"""Full evaluation: sklearn metrics, bootstrap CIs, reliability under corruption."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)

from src.calibration.metrics import (brier_score, classwise_ece, expected_calibration_error,
                                     mean_confidence, negative_log_likelihood, reliability_points)
from src.constants import ECE_BINS


def _numpy(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0):
    prob = torch.softmax(logits / max(0.05, temperature), dim=1).numpy()
    return prob, labels.numpy()


def summarize_full(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> dict:
    """One dict with every scalar metric. All values are JSON-safe floats."""
    scaled = logits / max(0.05, temperature)
    prob, y = _numpy(scaled, labels, 1.0)
    pred = prob.argmax(axis=1)
    out = {
        "acc": float(accuracy_score(y, pred)),
        "ece": expected_calibration_error(scaled, labels, ECE_BINS),
        "nll": negative_log_likelihood(scaled, labels),
        "brier": brier_score(scaled, labels),
        "meanconf": float(prob.max(axis=1).mean()),
        "classwise_ece": classwise_ece(scaled, labels, ECE_BINS),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "bal_acc": float(balanced_accuracy_score(y, pred)),
    }
    try:
        out["auroc_ovr"] = float(roc_auc_score(y, prob, multi_class="ovr"))
    except ValueError:
        out["auroc_ovr"] = float("nan")
    return out


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """95% percentile bootstrap CI for accuracy. Deterministic given seed."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = (y_pred[idx] == y_true[idx]).mean()
    return {"acc_ci95_lo": float(np.percentile(stats, 2.5)), "acc_ci95_hi": float(np.percentile(stats, 97.5))}


def confusion_list(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> list:
    prob, y = _numpy(logits, labels, temperature)
    return confusion_matrix(y, prob.argmax(axis=1)).tolist()


def reliability_full(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    return reliability_points(logits, labels, ECE_BINS)
