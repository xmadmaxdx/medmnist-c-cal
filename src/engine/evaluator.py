"""Clean and corrupted evaluation with calibration metrics."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.calibration.metrics import (brier_score, classwise_ece, expected_calibration_error,
                                     mean_confidence, negative_log_likelihood, reliability_points)
from src.constants import ECE_BINS
from src.data.corruptions import CORRUPTION_NAMES
from src.data.datasets import CorruptedWrapper
from src.utils.common import cleanup_cuda


@torch.no_grad()
def _collect(model, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_y: list[torch.Tensor] = []
    for x, y in loader:
        all_logits.append(model(x.to(device)).cpu())
        all_y.append(y)
    return torch.cat(all_logits), torch.cat(all_y)


def _summarize(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> dict:
    scaled = logits / max(0.05, temperature)
    prob = torch.softmax(scaled, dim=1)
    acc = float((prob.argmax(1) == labels).float().mean().item())
    return {
        "acc": acc,
        "ece": expected_calibration_error(scaled, labels, ECE_BINS),
        "nll": negative_log_likelihood(scaled, labels),
        "brier": brier_score(scaled, labels),
        "meanconf": mean_confidence(logits, temperature),
    }


def evaluate_clean(model, loader: DataLoader, device: torch.device, temperature: float = 1.0) -> dict:
    try:
        logits, labels = _collect(model, loader, device)
        return {"clean": _summarize(logits, labels, temperature)}
    finally:
        cleanup_cuda()


def evaluate_corrupted(model, test_images, test_labels, device, batch_size: int = 256, temperature: float = 1.0) -> dict:
    """Evaluate every corruption at severity 1-5. Returns nested dict."""
    from torch.utils.data import DataLoader as DL

    out: dict = {}
    try:
        for name in CORRUPTION_NAMES:
            out[name] = {}
            for sev in (1, 2, 3, 4, 5):
                ds = CorruptedWrapper(test_images, test_labels, name, sev)
                loader = DL(ds, batch_size=batch_size, shuffle=False, num_workers=0)
                logits, labels = _collect(model, loader, device)
                out[name][f"severity_{sev}"] = _summarize(logits, labels, temperature)
    finally:
        cleanup_cuda()
    return out
