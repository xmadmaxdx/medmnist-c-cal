"""Train entry point with validated config."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.config import load_config
from src.data.datasets import build_dataloaders
from src.losses.build import build_loss
from src.models.factory import build_model
from src.engine.trainer import train_model
from src.utils.common import setup_logging, set_seed, get_device
from src.utils.drive import ensure_drive_mounted, mirror_file

log = logging.getLogger("medmnist-c-cal")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train calibrated MedMNIST model")
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--smoke", action="store_true", help="Force 1-epoch fake-data smoke run")
    p.add_argument("--download", action="store_true", help="Allow MedMNIST download (Colab only)")
    return p.parse_args()


def main() -> None:
    setup_logging()
    ensure_drive_mounted()
    args = parse_args()
    cfg = load_config(args.config, smoke_override=args.smoke)
    set_seed(cfg.train.seed)
    device = get_device()
    log.info("device=%s config=%s", device, args.config)
    train_loader, val_loader, _ = build_dataloaders(cfg, download=args.download)
    model = build_model(cfg.model.name, cfg.model.num_classes).to(device)
    loss_fn = build_loss(cfg)
    history = train_model(cfg, model, loss_fn, train_loader, val_loader, device)
    if history.get("already_complete"):
        log.info("already complete, preserved %s", Path(cfg.train.checkpoint_dir) / "history.json")
        return
    out = Path(cfg.train.checkpoint_dir) / "history.json"
    out.write_text(json.dumps(history, indent=2), encoding="utf-8")
    mirror_file(out)
    log.info("saved %s best_val_acc=%.4f", out, history.get("best_val_acc", 0.0))


if __name__ == "__main__":
    main()
