"""Summarize *-full.json results: clean pre/post, seed stats, loss comparison."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="outputs/outputs_json")
    args = p.parse_args()
    files = sorted(glob.glob(f"{args.dir}/*-full.json"))
    rows = []
    for f in files:
        d = load(f)
        c0, c1 = d["clean_raw"], d["clean"]
        rows.append({
            "file": Path(f).stem,
            "acc_raw": round(c0["acc"], 4), "acc": round(c1["acc"], 4),
            "ece_raw": round(c0["ece"], 4), "ece": round(c1["ece"], 4),
            "nll_raw": round(c0["nll"], 4), "nll": round(c1["nll"], 4),
            "brier": round(c1["brier"], 4), "meanconf": round(c1["meanconf"], 4),
            "cw_ece": round(c1["classwise_ece"], 4),
            "macrof1": round(c1["macro_f1"], 4), "balacc": round(c1["bal_acc"], 4),
            "auroc": round(c1["auroc_ovr"], 4), "temp": round(d["temperature"], 4),
            "mean_corr_acc": round(d.get("mean_corrupted_acc", float("nan")), 4),
            "mean_corr_ece": round(d.get("mean_corrupted_ece", float("nan")), 4),
        })
    print(f"{'file':28s} {'acc':>7s} {'ece':>7s} {'nll':>7s} {'brier':>7s} {'mconf':>6s} {'cwece':>6s} {'macf1':>6s} {'balac':>6s} {'auroc':>6s} {'T':>6s} {'mca':>6s} {'mce':>6s}")
    for r in rows:
        print(f"{r['file']:28s} {r['acc']:7.4f} {r['ece']:7.4f} {r['nll']:7.4f} {r['brier']:7.4f} "
              f"{r['meanconf']:6.3f} {r['cw_ece']:6.3f} {r['macrof1']:6.3f} {r['balacc']:6.3f} "
              f"{r['auroc']:6.3f} {r['temp']:6.3f} {r['mean_corr_acc']:6.3f} {r['mean_corr_ece']:6.3f}")
    print("\nvs raw (TS delta acc / ece):")
    for r in rows:
        print(f"{r['file']:28s} dacc={r['acc']-r['acc_raw']:+.4f} dece={r['ece']-r['ece_raw']:+.4f} dnll={r['nll']-r['nll_raw']:+.4f}")


if __name__ == "__main__":
    main()
