"""Typed config loading with fail-closed validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "bloodmnist"
    root: str = "data"
    image_size: int = 28
    batch_size: int = 128
    num_workers: int = 2
    subset_train: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    name: str = "resnet18_small"
    pretrained: bool = False
    num_classes: int = 8
    dropout: float = 0.0


@dataclass(frozen=True)
class LossConfig:
    name: str = "ce"
    gamma: float = 2.0
    gamma2: float = 1.0
    eta: float = 1.0
    label_smoothing: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 5e-4
    optimizer: str = "adamw"
    seed: int = 0
    amp: bool = False
    checkpoint_dir: str = "outputs/checkpoints"
    smoke: bool = False


@dataclass(frozen=True)
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


_ALLOWED_DATASETS = {"bloodmnist", "dermamnist", "fake"}
_ALLOWED_MODELS = {"resnet18_small", "mobilenetv3_small", "simplecnn"}
_ALLOWED_LOSSES = {"ce", "focal", "dualfocal"}
_ALLOWED_OPTIMIZERS = {"adamw", "sgd"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_config(path: str | Path, smoke_override: bool = False) -> RunConfig:
    """Load YAML config with strict validation."""
    p = Path(path)
    _require(p.is_file(), f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    data = DataConfig(**raw.get("data", {}))
    model = ModelConfig(**raw.get("model", {}))
    loss = LossConfig(**raw.get("loss", {}))
    train = TrainConfig(**raw.get("train", {}))
    _require(data.dataset in _ALLOWED_DATASETS, f"bad data.dataset: {data.dataset}")
    _require(model.name in _ALLOWED_MODELS, f"bad model.name: {model.name}")
    _require(loss.name in _ALLOWED_LOSSES, f"bad loss.name: {loss.name}")
    _require(train.optimizer in _ALLOWED_OPTIMIZERS, f"bad optimizer: {train.optimizer}")
    _require(data.batch_size >= 1, "batch_size must be >= 1")
    _require(train.epochs >= 1, "epochs must be >= 1")
    _require(train.lr > 0, "lr must be > 0")
    _require(0.0 <= loss.label_smoothing < 1.0, "label_smoothing in [0,1)")
    _require(model.dropout == 0.0, "model.dropout unsupported, must be 0.0")
    _require(model.pretrained is False, "model.pretrained unsupported, must be false")
    _require(data.image_size == 28, "data.image_size unsupported, must be 28")
    _require(train.amp is False, "train.amp unsupported, must be false")
    _require(loss.name == "ce" or loss.label_smoothing == 0.0, "label_smoothing only for ce")
    _require(data.subset_train is None or data.subset_train >= 1, "subset_train must be >= 1")
    if smoke_override:
        train = TrainConfig(**{**train.__dict__, "smoke": True, "epochs": 1})
        data = DataConfig(**{**data.__dict__, "dataset": "fake", "subset_train": 200})
    return RunConfig(data=data, model=model, loss=loss, train=train)
