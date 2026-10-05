"""Focal loss with stable numerics."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import LOG_EPS


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float | None = None, reduction: str = "mean") -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1).clamp(min=-30.0)
        ce = F.nll_loss(logp, target, reduction="none")
        pt = (-ce).exp().clamp(min=LOG_EPS, max=1.0)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            loss = self.alpha * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
