"""Download MedMNIST npz files. Colab-only entry point."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Download BloodMNIST + DermaMNIST")
    p.add_argument("--root", default="data")
    p.add_argument("--sets", default="bloodmnist,dermamnist")
    args = p.parse_args()
    Path(args.root).mkdir(parents=True, exist_ok=True)
    from medmnist import BloodMNIST, DermaMNIST

    mapping = {"bloodmnist": BloodMNIST, "dermamnist": DermaMNIST}
    for name in [s.strip() for s in args.sets.split(",") if s.strip()]:
        if name not in mapping:
            raise ValueError(f"unknown set: {name}")
        for split in ("train", "val", "test"):
            mapping[name](split=split, download=True, size=28, root=args.root)
        print(f"ok: {name} -> {args.root}")


if __name__ == "__main__":
    main()
