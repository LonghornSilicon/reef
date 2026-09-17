"""MobileNetV3 built from the operator library."""

import torch
from torch import nn

from configs.classification.mobilenet_v3 import (
    MOBILENET_V3_CONFIGS,
    BneckRow,
    MobileNetV3Config,
)
from operators.activation import ReLU
from operators.convolution import Conv2d
from operators.dropout import Dropout
from operators.linear import Linear
from operators.normalization import BatchNorm2d
from operators.pooling import AdaptiveAvgPool2d

CHANNEL_DIVISOR = 8


class Hardswish(nn.Module):
    """``x * relu6(x + 3) / 6``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (x + 3.0).clamp(0.0, 6.0) / 6.0


class Hardsigmoid(nn.Module):
    """``relu6(x + 3) / 6``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 3.0).clamp(0.0, 6.0) / 6.0


def make_divisible(value: float, divisor: int = CHANNEL_DIVISOR) -> int:
    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    # Rounding down by more than 10% bumps up a step (TF-slim's rule).
    if rounded < 0.9 * value:
        rounded += divisor
    return rounded


class ConvNormActivation(nn.Sequential):
    """Conv2d, BatchNorm2d and an optional activation at indices 0, 1, 2."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        config: MobileNetV3Config,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        activation: type[nn.Module] | None = ReLU,
    ) -> None:
        layers: list[nn.Module] = [
            Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2 * dilation,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            BatchNorm2d(
                out_channels, eps=config.norm_eps, momentum=config.norm_momentum
            ),
        ]
        if activation is not None:
            layers.append(activation())
        super().__init__(*layers)


class SqueezeExcitation(nn.Module):
    """Channel gate from pooled features through two 1x1 convolutions."""

    def __init__(self, channels: int, squeeze_channels: int) -> None:
        super().__init__()
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc1 = Conv2d(channels, squeeze_channels, 1)
        self.fc2 = Conv2d(squeeze_channels, channels, 1)
        self.activation = ReLU()
        self.scale_activation = Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.activation(self.fc1(self.avgpool(x)))
        return self.scale_activation(self.fc2(scale)) * x


class InvertedResidual(nn.Module):
    """Expand, depthwise, optional squeeze-excite, project; residual if same."""

    def __init__(self, row: BneckRow, config: MobileNetV3Config) -> None:
        super().__init__()
        (
            in_channels,
            kernel,
            expanded,
            out_channels,
            use_se,
            activation,
            stride,
            dilation,
        ) = row
        act = Hardswish if activation == "HS" else ReLU
        self.use_res_connect = stride == 1 and in_channels == out_channels
        layers: list[nn.Module] = []
        if expanded != in_channels:
            layers.append(
                ConvNormActivation(
                    in_channels, expanded, 1, config, activation=act
                )
            )
        layers.append(
            ConvNormActivation(
                expanded,
                expanded,
                kernel,
                config,
                stride=1 if dilation > 1 else stride,
                dilation=dilation,
                groups=expanded,
                activation=act,
            )
        )
        if use_se:
            squeeze = make_divisible(expanded // 4)
            layers.append(SqueezeExcitation(expanded, squeeze))
        layers.append(
            ConvNormActivation(
                expanded, out_channels, 1, config, activation=None
            )
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return out + x if self.use_res_connect else out


class MobileNetV3(nn.Module):
    """Inverted-residual classifier matching torchvision layer for layer."""

    def __init__(self, config: MobileNetV3Config) -> None:
        super().__init__()
        self.config = config
        rows = config.inverted_residual_setting
        stem_channels = rows[0][0]
        last_conv_in = rows[-1][3]
        last_conv_out = 6 * last_conv_in
        self.features = nn.Sequential(
            ConvNormActivation(
                3, stem_channels, 3, config, stride=2, activation=Hardswish
            ),
            *(InvertedResidual(row, config) for row in rows),
            ConvNormActivation(
                last_conv_in, last_conv_out, 1, config, activation=Hardswish
            ),
        )
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            Linear(last_conv_out, config.last_channel),
            Hardswish(),
            Dropout(config.dropout),
            Linear(config.last_channel, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(self.features(x))
        return self.classifier(x.flatten(1))


def mobilenet_v3(key: str) -> MobileNetV3:
    if key not in MOBILENET_V3_CONFIGS:
        known = ", ".join(MOBILENET_V3_CONFIGS)
        raise KeyError(
            f"unknown MobileNetV3 size {key!r}; known sizes: {known}"
        )
    return MobileNetV3(MOBILENET_V3_CONFIGS[key])
