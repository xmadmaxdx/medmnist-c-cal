"""Training loop with checkpoint resume and cleanup."""
from __future__ import annotations

import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.common import cleanup_cuda
from src.utils.drive import mirror_file

log = logging.getLogger("medmnist-c-cal")


def _make_optimizer(cfg, model: torch.nn.Module):
    if cfg.train.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    return torch.optim.SGD(model.parameters(), lr=cfg.train.lr, momentum=0.9, weight_decay=cfg.train.weight_decay)


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == y).float().mean().item())


def train_model(cfg, model: torch.nn.Module, loss_fn, train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> dict:
    """Train and return history. Saves best checkpoint by val accuracy."""
    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"
    opt = _make_optimizer(cfg, model)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, cfg.train.epochs // 3), gamma=0.5)
    start_epoch = 0
    best_acc = 0.0
    if last_path.is_file():
        try:
            state = torch.load(last_path, map_location=device)
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["opt"])
            start_epoch = int(state.get("epoch", 0))
            best_acc = float(state.get("best_acc", 0.0))
            if "sched" in state:
                sched.load_state_dict(state["sched"])
            log.info("resumed from %s at epoch %d", last_path, start_epoch)
        except Exception as e:
            log.warning("resume failed, starting fresh: %s", e)
    if start_epoch >= cfg.train.epochs:
        log.info("already complete (%d/%d), preserving history.json", start_epoch, cfg.train.epochs)
        history = {"train_loss": [], "val_acc": [], "best_val_acc": best_acc, "already_complete": True}
        return history
    history: dict = {"train_loss": [], "val_acc": []}
    try:
        for epoch in range(start_epoch, cfg.train.epochs):
            model.train()
            total = 0.0
            n = 0
            for x, y in tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.train.epochs}", leave=False):
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite loss encountered")
                loss.backward()
                opt.step()
                total += float(loss.item()) * x.size(0)
                n += x.size(0)
            sched.step()
            model.eval()
            correct = 0
            total_v = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    correct += int((model(x).argmax(1) == y).sum().item())
                    total_v += x.size(0)
            acc = correct / max(1, total_v)
            history["train_loss"].append(total / max(1, n))
            history["val_acc"].append(acc)
            log.info("epoch=%d loss=%.4f val_acc=%.4f", epoch + 1, history["train_loss"][-1], acc)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch + 1, "best_acc": best_acc}, last_path)
            mirror_file(last_path)
            if acc > best_acc:
                best_acc = acc
                torch.save({"model": model.state_dict(), "val_acc": acc}, best_path)
                mirror_file(best_path)
    finally:
        cleanup_cuda()
    history["best_val_acc"] = best_acc
    return history
