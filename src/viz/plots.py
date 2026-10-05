"""Plotting: reliability diagrams and severity curves."""
from __future__ import annotations

import math
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#999999"]
_MARKERS = ["o", "s", "^", "D", "x", "+", "*", "p"]


def save_reliability(points: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    import math

    pairs = [(c, a) for c, a in zip(points["bin_conf"], points["bin_acc"]) if not (isinstance(a, float) and math.isnan(a))]
    xs = [c for c, _ in pairs]
    ys = [a for _, a in pairs]
    counts = [n for n, a in zip(points["bin_count"], points["bin_acc"]) if not (isinstance(a, float) and math.isnan(a))]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#0072B2", label="ideal")
    ax.plot(xs, ys, marker="o", markersize=4, linewidth=1, color="#E69F00", label="model")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return p


def save_reliability_counts(points: dict, path: str | Path) -> Path:
    """Reliability curve with bin counts on a twin axis. Sparse bins read as sparse."""
    import math

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pairs = [(c, a, n) for c, a, n in zip(points["bin_conf"], points["bin_acc"], points["bin_count"])
             if not (isinstance(a, float) and math.isnan(a))]
    xs = [c for c, _, _ in pairs]
    ys = [a for _, a, _ in pairs]
    ns = [n for _, _, n in pairs]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#0072B2", label="ideal")
    ax.plot(xs, ys, marker="o", markersize=4, linewidth=1, color="#E69F00", label="model")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax2 = ax.twinx()
    ax2.bar(xs, ns, width=0.04, color="#999999", alpha=0.45, label="count")
    ax2.set_ylabel("bin count")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return p


def save_severity_curve(severity_ece: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    # Stable order so curves keep styles across runs. ECE stored as fraction, plotted as percent.
    for i, name in enumerate(sorted(severity_ece)):
        vals = [100.0 * v for v in severity_ece[name]]
        ax.plot([1, 2, 3, 4, 5], vals, color=_COLORS[i % len(_COLORS)],
                marker=_MARKERS[i % len(_MARKERS)], markersize=4, linewidth=1, label=name)
    ax.set_xlabel("severity")
    ax.set_ylabel("ECE (%)")
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return p
