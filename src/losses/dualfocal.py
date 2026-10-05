"""Dual-Focal stable variant: focal on target plus complement penalty."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import LOG_EPS


class DualFocalLoss(nn.Module):
    """Stable Dual-Focal reimplementation for calibration study."""

    def __init__(self, gamma: float = 2.0, gamma2: float = 1.0, eta: float = 1.0) -> None:
        super().__init__()
        if gamma < 0 or gamma2 < 0 or eta < 0:
            raise ValueError("gamma, gamma2, eta must be >= 0")
        self.gamma = gamma
        self.gamma2 = gamma2
        self.eta = eta

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = F.softmax(logits, dim=1).clamp(min=LOG_EPS, max=1.0 - LOG_EPS)
        logp = prob.log()
        ce = F.nll_loss(logp, target, reduction="none")
        pt = prob.gather(1, target.unsqueeze(1)).squeeze(1).clamp(min=LOG_EPS, max=1.0)
        main = ((1.0 - pt) ** self.gamma) * ce
        # Complement term penalizes overconfident non-target mass.
        mask = torch.ones_like(prob).scatter_(1, target.unsqueeze(1), 0.0)
        comp = ((prob * mask).pow(self.gamma2) * (-(1.0 - prob + LOG_EPS).log()) * mask).sum(dim=1)
        return (main + self.eta * comp).mean()
