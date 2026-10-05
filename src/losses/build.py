"""Loss factory with typed contract."""
from __future__ import annotations

import torch.nn as nn

from src.losses.focal import FocalLoss
from src.losses.dualfocal import DualFocalLoss


def build_loss(cfg) -> nn.Module:
    name = cfg.loss.name
    if name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=cfg.loss.label_smoothing)
    if name == "focal":
        return FocalLoss(gamma=cfg.loss.gamma)
    if name == "dualfocal":
        return DualFocalLoss(gamma=cfg.loss.gamma, gamma2=cfg.loss.gamma2, eta=cfg.loss.eta)
    raise ValueError(f"unknown loss: {name}")
