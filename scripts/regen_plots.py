"""Regenerate severity plots from saved results JSONs. No data or GPU needed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.viz.plots import save_reliability_counts, save_severity_curve


def save_severity_acc(sev_acc: dict, path: Path) -> Path:
    from src.viz.plots import _COLORS, _MARKERS

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for i, name in enumerate(sorted(sev_acc)):
        ax.plot([1, 2, 3, 4, 5], [100.0 * v for v in sev_acc[name]], color=_COLORS[i % len(_COLORS)],
                marker=_MARKERS[i % len(_MARKERS)], markersize=4, linewidth=1, label=name)
    ax.set_xlabel("severity")
    ax.set_ylabel("accuracy (%)")
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Regen severity plots from JSON")
    p.add_argument("--json", required=True, help="results json with corrupted block")
    p.add_argument("--out", required=True, help="output png path (pdf twin auto-saved)")
    p.add_argument("--metric", default="ece", choices=["ece", "acc", "relcorr", "relclean"])
    p.add_argument("--corr", default="shot_noise", help="corruption key for relcorr mode")
    args = p.parse_args()
    results = json.loads(Path(args.json).read_text(encoding="utf-8"))
    out = Path(args.out)
    if args.metric == "relclean":
        save_reliability_counts(results["reliability_clean"], out)
    elif args.metric == "relcorr":
        rel = results.get("reliability_corrupted_sev3", {})
        if not rel:
            raise SystemExit(f"no corrupted reliability in {args.json}")
        if args.corr not in rel:
            raise SystemExit(f"corruption {args.corr!r} not in {args.json}")
        save_reliability_counts(rel[args.corr], out)
    else:
        corr = results.get("corrupted", {})
        if not corr or "note" in corr:
            raise SystemExit(f"no corrupted block in {args.json}")
        sev = {k: [v[f"severity_{s}"][args.metric] for s in (1, 2, 3, 4, 5)] for k, v in corr.items()}
        if args.metric == "ece":
            save_severity_curve(sev, out)
        else:
            save_severity_acc(sev, out)
    print(f"wrote {out} + {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
