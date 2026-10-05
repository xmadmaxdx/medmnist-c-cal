"""Calibration metrics: ECE, NLL, Brier."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def expected_calibration_error(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Equal-width ECE on confidence of predicted class."""
    if logits.numel() == 0:
        raise ValueError("empty logits")
    prob = F.softmax(logits, dim=1)
    conf, pred = prob.max(dim=1)
    acc = pred.eq(labels)
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=logits.device)
    ece = torch.zeros((), device=logits.device)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        n = int(mask.sum().item())
        if n > 0:
            ece = ece + (n / conf.numel()) * abs(float(acc[mask].float().mean()) - float(conf[mask].mean()))
    return float(ece.item())


def negative_log_likelihood(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(F.cross_entropy(logits, labels).item())


def brier_score(logits: torch.Tensor, labels: torch.Tensor) -> float:
    prob = F.softmax(logits, dim=1)
    onehot = F.one_hot(labels, num_classes=logits.size(1)).float()
    return float(((prob - onehot) ** 2).sum(dim=1).mean().item())


def reliability_points(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> dict:
    prob = F.softmax(logits, dim=1)
    conf, pred = prob.max(dim=1)
    acc = pred.eq(labels).float()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = {"bin_conf": [], "bin_acc": [], "bin_count": []}
    c = conf.detach().cpu().numpy()
    a = acc.detach().cpu().numpy()
    for i in range(n_bins):
        m = (c > edges[i]) & (c <= edges[i + 1]) if i > 0 else (c >= edges[i]) & (c <= edges[i + 1])
        out["bin_conf"].append(float(c[m].mean()) if m.sum() > 0 else float((edges[i] + edges[i + 1]) / 2))
        out["bin_acc"].append(float(a[m].mean()) if m.sum() > 0 else float("nan"))
        out["bin_count"].append(int(m.sum()))
    return out
