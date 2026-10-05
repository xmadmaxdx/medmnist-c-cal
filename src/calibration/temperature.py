"""Temperature scaling on held-out logits."""
from __future__ import annotations

import torch
import torch.nn as nn


def fit_temperature(val_logits: torch.Tensor, val_labels: torch.Tensor) -> float:
    """Fit scalar temperature with LBFGS on NLL."""
    if val_logits.numel() == 0:
        raise ValueError("empty validation logits")
    temp = nn.Parameter(torch.ones((), dtype=val_logits.dtype))
    opt = torch.optim.LBFGS([temp], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(val_logits / temp.clamp(min=0.05, max=10.0), val_labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(temp.clamp(min=0.05, max=10.0).detach().item())
