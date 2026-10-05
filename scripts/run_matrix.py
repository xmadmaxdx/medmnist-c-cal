"""Sweep 3 losses x 2 models on one dataset. Checkpointed per cell."""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path
import yaml


def main() -> None:
    p = argparse.ArgumentParser(description="Run calibration matrix")
    p.add_argument("--dataset", choices=["bloodmnist", "dermamnist"], default="bloodmnist")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--download", action="store_true")
    p.add_argument("--seeds", default="0", help="comma-separated seeds, ckpt dir suffixed per seed")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    if not seeds:
        raise SystemExit("empty --seeds")
    num_classes = 8 if args.dataset == "bloodmnist" else 7
    grid = list(itertools.product(["ce", "focal", "dualfocal"], ["resnet18_small", "simplecnn"]))
    tmp = Path("outputs/tmp-configs")
    tmp.mkdir(parents=True, exist_ok=True)
    for loss, model in grid:
      for seed in seeds:
        suffix = "" if (seed == 0 and len(seeds) == 1) else f"-s{seed}"
        cfg = {
            "data": {"dataset": args.dataset, "root": "data", "batch_size": 128, "num_workers": 2},
            "model": {"name": model, "num_classes": num_classes},
            "loss": {"name": loss, "gamma": 2.0, "gamma2": 1.0, "eta": 1.0},
            "train": {"epochs": args.epochs, "lr": 0.001, "optimizer": "adamw", "seed": seed,
                      "checkpoint_dir": f"outputs/ckpt-{args.dataset}-{model}-{loss}{suffix}"},
        }
        path = tmp / f"{args.dataset}-{model}-{loss}{suffix}.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        cmd = [sys.executable, "-m", "scripts.train", "--config", str(path)]
        if args.download:
            cmd.append("--download")
        print("RUN:", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, check=False)
        if r.returncode != 0:
            print(f"FAILED cell {loss}/{model}/seed{seed}, continuing", flush=True)


if __name__ == "__main__":
    main()
