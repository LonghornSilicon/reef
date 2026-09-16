"""AlexNet built from the from-scratch operator library."""

import torch
from torch import nn

from operators.activation import ReLU
from operators.convolution import Conv2d
from operators.dropout import Dropout
from operators.linear import Linear
from operators.pooling import AdaptiveAvgPool2d, MaxPool2d


class AlexNet(nn.Module):
    """Image classifier matching torchvision's AlexNet layer for layer."""

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        # Indices match torchvision's AlexNet, so its state_dict loads as-is.
        self.features = nn.Sequential(
            Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(64, 192, kernel_size=5, padding=2),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(192, 384, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(384, 256, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(256, 256, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            Dropout(p=dropout),
            Linear(256 * 6 * 6, 4096),
            ReLU(),
            Dropout(p=dropout),
            Linear(4096, 4096),
            ReLU(),
            Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)
