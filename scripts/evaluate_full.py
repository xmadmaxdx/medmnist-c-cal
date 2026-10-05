"""Extended evaluation: pre/post temperature, sklearn metrics, bootstrap, corruption reliability."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import torch

from src.config import load_config
from src.data.datasets import build_dataloaders
from src.models.factory import build_model
from src.calibration.temperature import fit_temperature
from src.engine.evaluator import evaluate_corrupted
from src.engine.full_metrics import (bootstrap_ci, confusion_list, reliability_full, summarize_full)
from src.utils.common import setup_logging, set_seed, get_device
from src.utils.drive import ensure_drive_mounted, mirror_file
from src.viz.plots import save_reliability, save_severity_curve

log = logging.getLogger("medmnist-c-cal")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full extended evaluation")
    p.add_argument("--config", required=True)
    p.add_argument("--download", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--boot", type=int, default=1000)
    return p.parse_args()


@torch.no_grad()
def _collect(model, loader, device):
    model.eval()
    ls, ys = [], []
    for x, y in loader:
        ls.append(model(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(ls), torch.cat(ys)


def main() -> None:
    setup_logging()
    ensure_drive_mounted()
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    device = get_device()
    _, val_loader, test_loader = build_dataloaders(cfg, download=args.download)
    model = build_model(cfg.model.name, cfg.model.num_classes).to(device)
    ckpt = Path(cfg.train.checkpoint_dir) / "best.pt"
    if not ckpt.is_file():
        ckpt = Path(cfg.train.checkpoint_dir) / "last.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"no checkpoint in {cfg.train.checkpoint_dir}; train first")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    vl, vy = _collect(model, val_loader, device)
    temp = fit_temperature(vl, vy)
    tl, ty = _collect(model, test_loader, device)
    clean_raw = summarize_full(tl, ty, 1.0)
    clean_cal = summarize_full(tl, ty, temp)
    clean_raw.update(bootstrap_ci(ty.numpy(), tl.argmax(1).numpy(), args.boot, cfg.train.seed))
    results: dict = {
        "temperature": temp,
        "clean_raw": clean_raw,
        "clean": clean_cal,
        "confusion": confusion_list(tl, ty, temp),
        "reliability_clean": reliability_full(tl / temp, ty),
    }
    if cfg.data.dataset == "fake":
        results["corrupted"] = {"note": "skipped for fake smoke"}
        sev_acc: dict = {}
        sev_ece: dict = {}
    else:
        from src.data.datasets import _load_medmnist_npz

        imgs, labels = _load_medmnist_npz(cfg.data.dataset, cfg.data.root, "test", args.download)
        corr = evaluate_corrupted(model, imgs, labels, device, cfg.data.batch_size, temp)
        detail: dict = {}
        sev_acc = {}
        sev_ece = {}
        for name, sevmap in corr.items():
            detail[name] = dict(sevmap)
            sev_acc[name] = [sevmap[f"severity_{s}"]["acc"] for s in (1, 2, 3, 4, 5)]
            sev_ece[name] = [sevmap[f"severity_{s}"]["ece"] for s in (1, 2, 3, 4, 5)]
        results["corrupted"] = detail
        results["mean_corrupted_acc"] = float(sum(v[4] for v in sev_acc.values()) / max(1, len(sev_acc)))
        results["mean_corrupted_ece"] = float(sum(v[4] for v in sev_ece.values()) / max(1, len(sev_ece)))
        from src.data.datasets import CorruptedWrapper
        from torch.utils.data import DataLoader as _DL

        rel_corr = {}
        for name in corr:
            ds = CorruptedWrapper(imgs, labels, name, 3)
            loader = _DL(ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=0)
            cl, cy = _collect(model, loader, device)
            rel_corr[name] = reliability_full(cl / temp, cy)
        results["reliability_corrupted_sev3"] = rel_corr
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    mirror_file(out)
    stem = out.stem
    for fig, tag in ((reliability_full(tl / temp, ty), "reliability"),):
        rp = save_reliability(fig, out.parent / f"{stem}.{tag}.png")
        mirror_file(rp)
        mirror_file(rp.with_suffix(".pdf"))
    if sev_acc:
        sp = save_severity_curve(sev_ece, out.parent / f"{stem}.severity_ece.png")
        mirror_file(sp)
        mirror_file(sp.with_suffix(".pdf"))
    log.info("wrote %s temp=%.3f", out, temp)


if __name__ == "__main__":
    main()
