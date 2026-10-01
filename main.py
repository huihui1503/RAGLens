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

ROOT_DIR = "./"
sys.path.append(os.path.join(ROOT_DIR, "src"))

from data_loading import RAGEvalDataset  # noqa: E402
from RAGLens import RAGLens  # noqa: E402
from sparsify import Sae  # noqa: E402

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
LLM_NAME = "meta-llama/Llama-2-7b-chat-hf"
SAE_NAME = "yuzhaouoe/Llama2-7b-SAE"
HOOKPOINT = "layers.15"
HF_CACHE_DIR = os.path.join(ROOT_DIR, "../huggingface/hub")
N_FOLDS = 4
N_REPEATS = 10  # repeat the 2-fold split with different shuffles
RANDOM_STATE = 0  # seeds the *sequence* of repeats, not a single split
PRED_THRESHOLD = 0.5  # for converting RAGLens scores -> binary labels


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_dolly_data(root_dir: str):
    data = RAGEvalDataset("Dolly", "test", root_dir=root_dir).items
    inputs = [item["input"] for item in data]
    outputs = [item["output"] for item in data]
    labels = [1 if len(item["hall_info"]) > 0 else 0 for item in data]
    return inputs, outputs, labels


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, cache_dir=HF_CACHE_DIR)

    model = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE_DIR,
        device_map="auto",
    )
    model.eval()

    sae = Sae.load_from_hub(SAE_NAME, hookpoint=HOOKPOINT, device="cuda")
    sae.cfg.transcode = "transcoder" in SAE_NAME
    sae.eval()

    return tokenizer, model, sae


# ----------------------------------------------------------------------
# Cross-validation
# ----------------------------------------------------------------------
def run_cross_validation(tokenizer, model, sae, inputs, outputs, labels):
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

        raglens = RAGLens(tokenizer=tokenizer, model=model, sae=sae, hookpoint=HOOKPOINT)
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


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    inputs, outputs, labels = load_dolly_data(ROOT_DIR)
    tokenizer, model, sae = load_models()
    metrics = run_cross_validation(tokenizer, model, sae, inputs, outputs, labels)
    print_summary(metrics)


if __name__ == "__main__":
    main()