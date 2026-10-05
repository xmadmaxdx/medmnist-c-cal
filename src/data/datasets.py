"""Dataset builders: real MedMNIST (Colab) + fake (local smoke)."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T

from src.data.corruptions import apply_corruption, CORRUPTION_NAMES


def train_transform() -> T.Compose:
    return T.Compose([T.RandomHorizontalFlip(p=0.5), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def test_transform() -> T.Compose:
    return T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class FakeClinical(Dataset):
    """Tiny synthetic RGB set for offline smoke tests."""

    def __init__(self, n: int = 200, num_classes: int = 8, transform=None, seed: int = 0) -> None:
        self.n = n
        self.num_classes = num_classes
        self.transform = transform
        rng = np.random.default_rng(seed)
        self.images = rng.integers(0, 256, size=(n, 28, 28, 3), dtype=np.uint8)
        self.labels = rng.integers(0, num_classes, size=(n,))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        img = Image.fromarray(self.images[i])
        if self.transform is not None:
            return self.transform(img), int(self.labels[i])
        return T.ToTensor()(img), int(self.labels[i])


class _NpzWrapper(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None) -> None:
        self.images = images
        self.labels = labels.reshape(-1)
        self.transform = transform or test_transform()

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, i: int):
        img = Image.fromarray(self.images[i])
        return self.transform(img), int(self.labels[i])


class CorruptedWrapper(Dataset):
    """Wrap clean test images with one corruption at fixed severity."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, corruption: str, severity: int) -> None:
        if corruption not in CORRUPTION_NAMES:
            raise ValueError(f"unknown corruption: {corruption}")
        self.images = images
        self.labels = labels.reshape(-1)
        self.corruption = corruption
        self.severity = severity
        self.transform = test_transform()

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, i: int):
        base = Image.fromarray(self.images[i])
        corr = apply_corruption(base, self.corruption, self.severity, seed=int(i))
        return self.transform(corr), int(self.labels[i])


def _load_medmnist_npz(dataset: str, root: str, split: str, download: bool) -> tuple[np.ndarray, np.ndarray]:
    """Load via medmnist API. Only called on Colab with download=True."""
    from medmnist import BloodMNIST, DermaMNIST

    mapping = {"bloodmnist": BloodMNIST, "dermamnist": DermaMNIST}
    if dataset not in mapping:
        raise ValueError(f"unsupported dataset: {dataset}")
    Path(root).mkdir(parents=True, exist_ok=True)
    cls = mapping[dataset]
    ds = cls(split=split, download=download, size=28, root=root)
    imgs = np.asarray(ds.imgs)
    labels = np.asarray(ds.labels)
    if imgs.ndim == 3:
        imgs = np.stack([imgs, imgs, imgs], axis=-1)
    return imgs, labels


def build_dataloaders(cfg, download: bool = False) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test loaders with fail-closed validation."""
    name = cfg.data.dataset
    if name == "fake":
        tr = FakeClinical(n=cfg.data.subset_train or 200, num_classes=cfg.model.num_classes, transform=train_transform(), seed=0)
        va = FakeClinical(n=64, num_classes=cfg.model.num_classes, transform=test_transform(), seed=1)
        te = FakeClinical(n=64, num_classes=cfg.model.num_classes, transform=test_transform(), seed=2)
    else:
        tr_imgs, tr_labels = _load_medmnist_npz(name, cfg.data.root, "train", download)
        va_imgs, va_labels = _load_medmnist_npz(name, cfg.data.root, "val", download)
        te_imgs, te_labels = _load_medmnist_npz(name, cfg.data.root, "test", download)
        if cfg.data.subset_train is not None:
            k = min(int(cfg.data.subset_train), int(tr_imgs.shape[0]))
            tr_imgs, tr_labels = tr_imgs[:k], tr_labels[:k]
        tr = _NpzWrapper(tr_imgs, tr_labels, train_transform())
        va = _NpzWrapper(va_imgs, va_labels, test_transform())
        te = _NpzWrapper(te_imgs, te_labels, test_transform())
    kw = {"batch_size": cfg.data.batch_size, "num_workers": cfg.data.num_workers, "pin_memory": torch.cuda.is_available()}
    return (
        DataLoader(tr, shuffle=True, **kw),
        DataLoader(va, shuffle=False, **kw),
        DataLoader(te, shuffle=False, **kw),
    )
