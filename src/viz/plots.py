"""Plotting: reliability diagrams and severity curves."""
from __future__ import annotations

import math
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_reliability(points: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    xs = [c for c, a in zip(points["bin_conf"], points["bin_acc"]) if not math.isnan(a)]
    ys = [a for a in points["bin_acc"] if not math.isnan(a)]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.plot(xs, ys, marker="o", linewidth=1)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def save_severity_curve(severity_ece: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, vals in severity_ece.items():
        ax.plot([1, 2, 3, 4, 5], vals, marker="o", label=name, linewidth=1)
    ax.set_xlabel("severity")
    ax.set_ylabel("ECE")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p
