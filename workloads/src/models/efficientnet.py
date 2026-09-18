"""EfficientNet built from the operator library."""

import math

import torch
from torch import nn

from configs.efficientnet import (
    EFFICIENTNET_CONFIGS,
    EfficientNetConfig,
)
from operators.activation import Sigmoid, SiLU
from operators.convolution import Conv2d
from operators.dropout import Dropout
from operators.linear import Linear
from operators.normalization import BatchNorm2d
from operators.pooling import AdaptiveAvgPool2d


def make_divisible(value: float, divisor: int = 8) -> int:
    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    # torchvision's _make_divisible never rounds down by more than 10%.
    if rounded < 0.9 * value:
        rounded += divisor
    return rounded


def conv_norm_act(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    config: EfficientNetConfig,
    stride: int = 1,
    groups: int = 1,
    activation: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False,
        ),
        BatchNorm2d(
            out_channels, eps=config.norm_eps, momentum=config.norm_momentum
        ),
    ]
    if activation:
        layers.append(SiLU())
    return nn.Sequential(*layers)


class SqueezeExcitation(nn.Module):
    """Channel gate from a pooled 1x1 bottleneck, as in torchvision.ops."""

    def __init__(self, channels: int, squeeze_channels: int) -> None:
        super().__init__()
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc1 = Conv2d(channels, squeeze_channels, 1)
        self.fc2 = Conv2d(squeeze_channels, channels, 1)
        self.activation = SiLU()
        self.scale_activation = Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc2(self.activation(self.fc1(self.avgpool(x))))
        return self.scale_activation(scale) * x


class MBConv(nn.Module):
    """Inverted residual: expand 1x1, depthwise kxk, squeeze-excite, project."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        kernel_size: int,
        stride: int,
        config: EfficientNetConfig,
    ) -> None:
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        expanded = make_divisible(in_channels * expand_ratio)
        layers: list[nn.Module] = []
        if expanded != in_channels:
            layers.append(conv_norm_act(in_channels, expanded, 1, config))
        layers.append(
            conv_norm_act(
                expanded,
                expanded,
                kernel_size,
                config,
                stride=stride,
                groups=expanded,
            )
        )
        layers.append(SqueezeExcitation(expanded, max(1, in_channels // 4)))
        layers.append(
            conv_norm_act(expanded, out_channels, 1, config, activation=False)
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return out + x if self.use_res_connect else out


class FusedMBConv(nn.Module):
    """MBConv with the expand and depthwise convolutions fused into one kxk."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        kernel_size: int,
        stride: int,
        config: EfficientNetConfig,
    ) -> None:
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        expanded = make_divisible(in_channels * expand_ratio)
        if expanded != in_channels:
            layers = [
                conv_norm_act(
                    in_channels, expanded, kernel_size, config, stride=stride
                ),
                conv_norm_act(
                    expanded, out_channels, 1, config, activation=False
                ),
            ]
        else:
            layers = [
                conv_norm_act(
                    in_channels, out_channels, kernel_size, config, stride
                )
            ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return out + x if self.use_res_connect else out


BLOCKS: dict[str, type[MBConv | FusedMBConv]] = {
    "mbconv": MBConv,
    "fused": FusedMBConv,
}


class EfficientNet(nn.Module):
    """Compound-scaled MBConv classifier matching torchvision layer by layer."""

    def __init__(self, config: EfficientNetConfig) -> None:
        super().__init__()
        self.config = config
        rows = config.inverted_residual_setting
        stem_channels = make_divisible(rows[0][3] * config.width_mult)
        layers: list[nn.Module] = [
            conv_norm_act(3, stem_channels, 3, config, stride=2)
        ]
        out_channels = stem_channels
        for expand, kernel, stride, in_base, out_base, count, kind in rows:
            block = BLOCKS[kind]
            in_channels = make_divisible(in_base * config.width_mult)
            out_channels = make_divisible(out_base * config.width_mult)
            depth = math.ceil(count * config.depth_mult)
            stage = [
                block(in_channels, out_channels, expand, kernel, stride, config)
            ]
            stage.extend(
                block(out_channels, out_channels, expand, kernel, 1, config)
                for _ in range(1, depth)
            )
            layers.append(nn.Sequential(*stage))
        last_channel = config.last_channel or 4 * out_channels
        layers.append(conv_norm_act(out_channels, last_channel, 1, config))
        self.features = nn.Sequential(*layers)
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            Dropout(config.dropout), Linear(last_channel, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(self.features(x))
        return self.classifier(x.flatten(1))


def efficientnet(key: str) -> EfficientNet:
    if key not in EFFICIENTNET_CONFIGS:
        known = ", ".join(EFFICIENTNET_CONFIGS)
        raise KeyError(
            f"unknown EfficientNet size {key!r}; known sizes: {known}"
        )
    return EfficientNet(EFFICIENTNET_CONFIGS[key])
