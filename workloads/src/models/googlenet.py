"""GoogLeNet built from the operator library."""

import torch
from torch import nn

from configs.googlenet import GOOGLENET_CONFIGS, GoogLeNetConfig, InceptionSpec
from operators.activation import ReLU
from operators.convolution import Conv2d
from operators.dropout import Dropout
from operators.linear import Linear
from operators.normalization import BatchNorm2d
from operators.pooling import AdaptiveAvgPool2d, MaxPool2d

AUX_POOLED_SIZE = 4
AUX_CHANNELS = 128
AUX_HIDDEN = 1024


class BasicConv2d(nn.Module):
    """Bias-free convolution, batch norm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        norm_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_channels, eps=norm_eps)
        self.relu = ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Inception(nn.Module):
    """Four parallel branches concatenated along channels."""

    def __init__(self, spec: InceptionSpec, norm_eps: float) -> None:
        super().__init__()
        in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj = spec
        self.branch1 = BasicConv2d(in_channels, ch1x1, 1, norm_eps=norm_eps)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, 1, norm_eps=norm_eps),
            BasicConv2d(ch3x3red, ch3x3, 3, padding=1, norm_eps=norm_eps),
        )
        # torchvision's "5x5" branch is really 3x3 (pytorch/vision#906); the
        # published checkpoint depends on it.
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, 1, norm_eps=norm_eps),
            BasicConv2d(ch5x5red, ch5x5, 3, padding=1, norm_eps=norm_eps),
        )
        self.branch4 = nn.Sequential(
            MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, 1, norm_eps=norm_eps),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = (
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
        )
        return torch.cat(branches, dim=1)


class InceptionAux(nn.Module):
    """Auxiliary classifier hung off a mid-network feature map."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.avgpool = AdaptiveAvgPool2d((AUX_POOLED_SIZE, AUX_POOLED_SIZE))
        self.conv = BasicConv2d(in_channels, AUX_CHANNELS, 1, norm_eps=norm_eps)
        self.fc1 = Linear(AUX_CHANNELS * AUX_POOLED_SIZE**2, AUX_HIDDEN)
        self.relu = ReLU()
        self.dropout = Dropout(dropout)
        self.fc2 = Linear(AUX_HIDDEN, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.avgpool(x)).flatten(1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


class GoogLeNet(nn.Module):
    """Inception v1 classifier matching torchvision layer for layer."""

    def __init__(self, config: GoogLeNetConfig) -> None:
        super().__init__()
        self.config = config
        eps = config.norm_eps
        self.conv1 = BasicConv2d(3, 64, 7, stride=2, padding=3, norm_eps=eps)
        self.maxpool1 = MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, 1, norm_eps=eps)
        self.conv3 = BasicConv2d(64, 192, 3, padding=1, norm_eps=eps)
        self.maxpool2 = MaxPool2d(3, stride=2, ceil_mode=True)
        blocks = [Inception(spec, eps) for spec in config.inception_blocks]
        (
            self.inception3a,
            self.inception3b,
            self.inception4a,
            self.inception4b,
            self.inception4c,
            self.inception4d,
            self.inception4e,
            self.inception5a,
            self.inception5b,
        ) = blocks
        self.maxpool3 = MaxPool2d(3, stride=2, ceil_mode=True)
        self.maxpool4 = MaxPool2d(2, stride=2, ceil_mode=True)
        self.aux1 = self.aux2 = None
        if config.aux_logits:
            aux_in_4a = config.inception_blocks[3][0]
            aux_in_4d = config.inception_blocks[6][0]
            self.aux1 = InceptionAux(
                aux_in_4a, config.num_classes, config.dropout_aux, eps
            )
            self.aux2 = InceptionAux(
                aux_in_4d, config.num_classes, config.dropout_aux, eps
            )
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(config.dropout)
        # Width of 5b's concat: ch1x1 + ch3x3 + ch5x5 + pool_proj.
        final_channels = sum(
            config.inception_blocks[-1][i] for i in (1, 3, 5, 6)
        )
        self.fc = Linear(final_channels, config.num_classes)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # No transform_input: torchvision turns it on only for its pretrained
        # weights, so build the reference with the default (False).
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv3(self.conv2(x)))
        x = self.maxpool3(self.inception3b(self.inception3a(x)))
        x = self.inception4a(x)
        aux1 = self.aux1(x) if self.training and self.aux1 is not None else None
        x = self.inception4d(self.inception4c(self.inception4b(x)))
        aux2 = self.aux2(x) if self.training and self.aux2 is not None else None
        x = self.maxpool4(self.inception4e(x))
        x = self.inception5b(self.inception5a(x))
        logits = self.fc(self.dropout(self.avgpool(x).flatten(1)))
        if aux1 is None or aux2 is None:
            return logits
        return logits, aux2, aux1


def googlenet(key: str) -> GoogLeNet:
    if key not in GOOGLENET_CONFIGS:
        known = ", ".join(GOOGLENET_CONFIGS)
        raise KeyError(f"unknown GoogLeNet size {key!r}; known sizes: {known}")
    return GoogLeNet(GOOGLENET_CONFIGS[key])
