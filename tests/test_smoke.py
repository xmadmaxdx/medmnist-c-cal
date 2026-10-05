"""Fast offline smoke tests. No downloads, no GPU required."""
from __future__ import annotations

import torch

from src.calibration.metrics import expected_calibration_error, brier_score
from src.calibration.temperature import fit_temperature
from src.config import load_config
from src.data.corruptions import apply_corruption, CORRUPTION_NAMES
from src.data.datasets import FakeClinical, build_dataloaders
from src.losses.build import build_loss
from src.losses.dualfocal import DualFocalLoss
from src.losses.focal import FocalLoss
from src.models.factory import build_model
from PIL import Image
import numpy as np


def test_config_loads() -> None:
    cfg = load_config("configs/smoke.yaml")
    assert cfg.model.num_classes == 8


def test_fake_forward() -> None:
    cfg = load_config("configs/smoke.yaml")
    tr, _, _ = build_dataloaders(cfg)
    x, y = next(iter(tr))
    m = build_model(cfg.model.name, cfg.model.num_classes)
    out = m(x)
    assert out.shape[0] == x.shape[0]


def test_losses_finite() -> None:
    logits = torch.randn(16, 8)
    target = torch.randint(0, 8, (16,))
    for fn in (FocalLoss(), DualFocalLoss()):
        assert torch.isfinite(fn(logits, target))


def test_corruptions_run() -> None:
    img = Image.fromarray(np.random.default_rng(1).integers(0, 256, (28, 28, 3), dtype=np.uint8))
    for name in CORRUPTION_NAMES:
        out = apply_corruption(img, name, 3, seed=0)
        assert out.size == (28, 28)


def test_calibration() -> None:
    logits = torch.randn(64, 8)
    labels = torch.randint(0, 8, (64,))
    assert 0.0 <= expected_calibration_error(logits, labels) <= 1.0
    assert brier_score(logits, labels) >= 0.0
    assert 0.05 <= fit_temperature(logits, labels) <= 10.0
