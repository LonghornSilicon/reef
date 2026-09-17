"""VGG built from the operator library."""

import torch
from torch import nn

from configs.vgg import VGG_CONFIGS, VGGConfig
from operators.activation import ReLU
from operators.convolution import Conv2d
from operators.dropout import Dropout
from operators.linear import Linear
from operators.normalization import BatchNorm2d
from operators.pooling import AdaptiveAvgPool2d, MaxPool2d

POOLED_SIZE = 7


def make_features(config: VGGConfig) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels = 3
    for spec in config.layers:
        if spec == "M":
            layers.append(MaxPool2d(kernel_size=2, stride=2))
            continue
        out_channels = int(spec)
        layers.append(Conv2d(in_channels, out_channels, 3, padding=1))
        if config.batch_norm:
            layers.append(BatchNorm2d(out_channels))
        layers.append(ReLU())
        in_channels = out_channels
    return nn.Sequential(*layers)


class VGG(nn.Module):
    """Plain conv stack, 7x7 average pool and three-layer classifier."""

    def __init__(self, config: VGGConfig) -> None:
        super().__init__()
        self.config = config
        self.features = make_features(config)
        self.avgpool = AdaptiveAvgPool2d((POOLED_SIZE, POOLED_SIZE))
        last_channels = [c for c in config.layers if isinstance(c, int)][-1]
        self.classifier = nn.Sequential(
            Linear(last_channels * POOLED_SIZE * POOLED_SIZE, 4096),
            ReLU(),
            Dropout(config.dropout),
            Linear(4096, 4096),
            ReLU(),
            Dropout(config.dropout),
            Linear(4096, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(self.features(x))
        return self.classifier(x.flatten(1))


def vgg(key: str) -> VGG:
    if key not in VGG_CONFIGS:
        known = ", ".join(VGG_CONFIGS)
        raise KeyError(f"unknown VGG size {key!r}; known sizes: {known}")
    return VGG(VGG_CONFIGS[key])
