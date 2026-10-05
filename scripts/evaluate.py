"""Evaluate clean + corrupted ECE/NLL/Brier with temperature scaling."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import torch

from src.config import load_config
from src.data.datasets import build_dataloaders
from src.models.factory import build_model
from src.calibration.metrics import reliability_points
from src.calibration.temperature import fit_temperature
from src.engine.evaluator import evaluate_clean, evaluate_corrupted
from src.utils.common import setup_logging, set_seed, get_device
from src.utils.drive import ensure_drive_mounted, mirror_file
from src.viz.plots import save_reliability, save_severity_curve

log = logging.getLogger("medmnist-c-cal")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate calibration under corruption")
    p.add_argument("--config", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--download", action="store_true")
    p.add_argument("--out", default="outputs/results.json")
    return p.parse_args()


@torch.no_grad()
def _val_logits(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
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
    cfg = load_config(args.config, smoke_override=args.smoke)
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
    vl, vy = _val_logits(model, val_loader, device)
    temp = fit_temperature(vl, vy)
    clean = evaluate_clean(model, test_loader, device, temp)
    log.info("clean=%s temp=%.3f", clean, temp)
    # Corrupted eval needs raw test arrays; use loader dataset when fake, else reload npz.
    results: dict = {"temperature": temp, **clean}
    if cfg.data.dataset == "fake":
        results["corrupted"] = {"note": "skipped for fake smoke"}
        sev_curve = {}
    else:
        from src.data.datasets import _load_medmnist_npz

        imgs, labels = _load_medmnist_npz(cfg.data.dataset, cfg.data.root, "test", args.download)
        corr = evaluate_corrupted(model, imgs, labels, device, cfg.data.batch_size, temp)
        results["corrupted"] = corr
        sev_curve = {k: [v[f"severity_{s}"]["ece"] for s in (1, 2, 3, 4, 5)] for k, v in corr.items()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    mirror_file(out)
    tl, ty = _val_logits(model, test_loader, device)
    stem = out.stem
    for ext in ("png", "pdf"):
        rp = save_reliability(reliability_points(tl / temp, ty), out.parent / f"{stem}.reliability.{ext}")
        mirror_file(rp)
    if sev_curve:
        for ext in ("png", "pdf"):
            sp = save_severity_curve(sev_curve, out.parent / f"{stem}.severity_ece.{ext}")
            mirror_file(sp)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
