"""
Train and evaluate RAGLens on the Dolly (Accurate Context) dataset
using 2-fold stratified cross-validation, following Xiong et al. (2026).
"""

import os
import sys

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from sae_lens import SAE

ROOT_DIR = "./"
sys.path.append(os.path.join(ROOT_DIR, "src"))

from data_loading import RAGEvalDataset  # noqa: E402
from RAGLens import RAGLens  # noqa: E402
from sparsify import Sae  # noqa: E402
import argparse
from huggingface_hub import snapshot_download

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
HF_CACHE_DIR = os.path.join(ROOT_DIR, "../huggingface/hub")
N_FOLDS = 4
N_REPEATS = 10  # repeat the 2-fold split with different shuffles
RANDOM_STATE = 0  # seeds the *sequence* of repeats, not a single split
PRED_THRESHOLD = 0.5  # for converting RAGLens scores -> binary labels

# ----------------------------------------------------------------------
# Parameter loading
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run hallucination-detection pipeline")
    parser.add_argument(
        "-model", "--model",
        type=str,
        default=None,
        choices=["llama2-7b", "llama2-13b", "mistral-7b", "llama3-8b"],
        help="Which model to run"
    )
    parser.add_argument(
        "-dataset", "--dataset",
        type=str,
        default=None,
        choices=["ragtruth", "hallurag", "dolly"],
        help="Quantization mode (optional)"
    )

    return parser.parse_args()

# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_data(dataset_name: str, llm_model: str):
    data_dir = f"./dataset/{dataset_name}/merged.jsonl"
    def load_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    data = load_jsonl(data_dir)
    inputs_train, outputs_train, labels_train  = [], [], []
    inputs_test, outputs_test, labels_test  = [], [], []
    for i in data:
        if i["model"] == llm_model:
            if i["split"] == "train":
                inputs_train.append(i["prompt"])
                outputs_train.append(i["response"])
                labels_train.append(1 if len(i["labels"]) > 0 else 0)
            elif i["split"] == "test":
                inputs_test.append(i["prompt"])
                outputs_test.append(i["response"])
                labels_test.append(1 if len(i["labels"]) > 0 else 0)
    return inputs_train, outputs_train, labels_train, inputs_test, outputs_test, labels_test

# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
def load_models(llm_model_name, sae_model_name, hook_point):
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name, cache_dir=HF_CACHE_DIR)

    model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE_DIR,
        device_map="auto",
    )
    model.eval()

    if llm_model_name == "mistralai/Mistral-7B-Instruct-v0.1":
        sae = SAE.load_from_disk(sae_model_name, device="cuda")
    else:
        sae = Sae.load_from_hub(sae_model_name, hookpoint=hook_point, device="cuda")

    sae.cfg.transcode = "transcoder" in sae_model_name
    sae.eval()

    return tokenizer, model, sae


# ----------------------------------------------------------------------
# Cross-validation
# ----------------------------------------------------------------------
def run_cross_validation(tokenizer, model, sae, inputs, outputs, labels, hook_point):
    """
    Repeated 2-fold stratified CV: the 2-fold split is repeated N_REPEATS
    times with different shuffles, so the final estimate isn't tied to
    any single random_state / partition of the data.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )

    metrics = {"auroc": [], "balanced_acc": [], "macro_f1": []}

    for split_num, (train_idx, test_idx) in enumerate(cv.split(outputs, labels), start=1):
        repeat_num = (split_num - 1) // N_FOLDS + 1
        fold_num = (split_num - 1) % N_FOLDS + 1

        inputs_train = [inputs[i] for i in train_idx]
        inputs_test = [inputs[i] for i in test_idx]
        outputs_train = [outputs[i] for i in train_idx]
        outputs_test = [outputs[i] for i in test_idx]
        y_train = [labels[i] for i in train_idx]
        y_test = [labels[i] for i in test_idx]

        raglens = RAGLens(tokenizer=tokenizer, model=model, sae=sae, hookpoint=hook_point)
        raglens.fit(inputs=inputs_train, outputs=outputs_train, labels=y_train)

        logits = raglens.predict_proba(inputs=inputs_test, outputs=outputs_test)
        preds = (logits > 0.5).astype(int)

        split_auroc = roc_auc_score(y_test, logits)
        split_balanced_acc = balanced_accuracy_score(y_test, preds)
        split_macro_f1 = f1_score(y_test, preds, average="macro")

        metrics["auroc"].append(split_auroc)
        metrics["balanced_acc"].append(split_balanced_acc)
        metrics["macro_f1"].append(split_macro_f1)

        print(
            f"[Repeat {repeat_num}/{N_REPEATS} Fold {fold_num}/{N_FOLDS}] "
            f"AUROC={split_auroc:.4f}  BalancedAcc={split_balanced_acc:.4f}  "
            f"MacroF1={split_macro_f1:.4f}"
        )

    return metrics


def print_summary(metrics: dict):
    n_splits = N_FOLDS * N_REPEATS
    print(f"\n=== Summary (mean ± std across {n_splits} splits, {N_REPEATS} repeats) ===")
    for name, values in metrics.items():
        print(f"{name}: {np.mean(values):.4f} ± {np.std(values):.4f}")

def evaluate_on_test_data(
        tokenizer,
        model,
        sae,
        hook_point,
        inputs_train,
        outputs_train,
        y_train,
        inputs_test,
        outputs_test,
        y_test
        ):
    raglens = RAGLens(tokenizer=tokenizer, model=model, sae=sae, hookpoint=hook_point)
    raglens.fit(inputs=inputs_train, outputs=outputs_train, labels=y_train)

    logits = raglens.predict_proba(inputs=inputs_test, outputs=outputs_test)
    preds = (logits > 0.5).astype(int)

    auroc = roc_auc_score(y_test, logits)
    balanced_acc = balanced_accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    print(
        f"AUROC={auroc:.4f}  BalancedAcc={balanced_acc:.4f}  "
        f"MacroF1={macro_f1:.4f}"
    )
    
# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    print(args.model, args.dataset)
    if args.model == "llama2-7b":
        llm_model_name = "meta-llama/Llama-2-7b-chat-hf"
        sae_model_name = "yuzhaouoe/Llama2-7b-SAE"
        data_type = "llama-2-7b-chat"
        hook_point = "layers.15"
    elif args.model == "llama2-13b":
        llm_model_name = "meta-llama/Llama-2-13b-chat-hf"
        sae_model_name = "gzxiong/sae-llama-2-13b-chat"
        hook_point = "layers.15"
        data_type="llama-2-13b-chat"
    elif args.model == "llama3-8b":
        llm_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        sae_model_name = "EleutherAI/sae-llama-3-8b-32x"
        hook_point = "layers.15"
        data_type = "llama-3-8b-instruct"
    elif args.model == "mistral-7b":
        llm_model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        # sae_model_name = "/home/huy/baselines/RAGLens/mistral-7b-sparse-autoencoder-layer16"
        sae_model_name = "/home/huy/baselines/RAGLens/saes/mistral-layer16"
        hook_point = "layers.16"
        data_type = "mistral-7B-instruct"
    else:
        print("model name error")
        exit(-1)


    inputs_train, outputs_train, labels_train, inputs_test, outputs_test, labels_test = load_data(
        dataset_name=args.dataset,
        llm_model=data_type,
    )

    tokenizer, llm_model, sae_model = load_models(
        llm_model_name=llm_model_name,
        sae_model_name=sae_model_name,
        hook_point=hook_point,
    )

    if len(inputs_train) == 0:
        print(f"Run cross validation on {args.model} and {args.dataset}")
        metrics = run_cross_validation(tokenizer, llm_model, sae_model, inputs_test, outputs_test, labels_test, hook_point)
        print_summary(metrics)
    else:
        print(f"Run evaluation on {args.model} and {args.dataset}")
        evaluate_on_test_data(
            tokenizer = tokenizer,
            model = llm_model,
            sae = sae_model,
            hook_point = hook_point,
            inputs_train = inputs_train,
            outputs_train = outputs_train,
            y_train = labels_train,
            inputs_test = inputs_test,
            outputs_test = outputs_test,
            y_test = labels_test
            )
        


if __name__ == "__main__":
    main()