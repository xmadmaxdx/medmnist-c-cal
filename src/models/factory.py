"""Small-image backbones with explicit contracts."""
from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvm


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


def _resnet18_small(num_classes: int) -> nn.Module:
    m = tvm.resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def _mobilenet_small(num_classes: int) -> nn.Module:
    m = tvm.mobilenet_v3_small(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


def build_model(name: str, num_classes: int) -> nn.Module:
    """Build backbone. Raises on unknown name."""
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    if name == "resnet18_small":
        return _resnet18_small(num_classes)
    if name == "mobilenetv3_small":
        return _mobilenet_small(num_classes)
    if name == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    raise ValueError(f"unknown model: {name}")
