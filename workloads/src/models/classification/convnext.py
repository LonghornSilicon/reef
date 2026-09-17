"""ConvNeXt built from the operator library."""

import torch
from torch import nn

from configs.classification.convnext import CONVNEXT_CONFIGS, ConvNeXtConfig
from operators.activation import ErfGELU
from operators.convolution import Conv2d
from operators.linear import Linear
from operators.normalization import LayerNorm
from operators.pooling import AdaptiveAvgPool2d


class Permute(nn.Module):
    """Parameter-free axis reordering, as torchvision.ops.Permute."""

    def __init__(self, dims: tuple[int, ...]) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)


class LayerNorm2d(LayerNorm):
    """LayerNorm over the channel axis of a channels-first feature map."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class CNBlock(nn.Module):
    """Depthwise 7x7, channels-last MLP, layer scale and a residual."""

    def __init__(self, dim: int, layer_scale: float, norm_eps: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Conv2d(dim, dim, 7, padding=3, groups=dim),
            Permute((0, 2, 3, 1)),
            LayerNorm(dim, eps=norm_eps),
            Linear(dim, 4 * dim),
            ErfGELU(),
            Linear(4 * dim, dim),
            Permute((0, 3, 1, 2)),
        )
        self.layer_scale = nn.Parameter(torch.full((dim, 1, 1), layer_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layer_scale * self.block(x)


class ConvNeXt(nn.Module):
    """Patchified ConvNet classifier matching torchvision layer for layer."""

    def __init__(self, config: ConvNeXtConfig) -> None:
        super().__init__()
        self.config = config
        eps = config.norm_eps
        stem_channels = config.block_setting[0][0]
        layers: list[nn.Module] = [
            nn.Sequential(
                Conv2d(3, stem_channels, 4, stride=4),
                LayerNorm2d(stem_channels, eps=eps),
            )
        ]
        for in_channels, out_channels, count in config.block_setting:
            layers.append(
                nn.Sequential(
                    *(
                        CNBlock(in_channels, config.layer_scale, eps)
                        for _ in range(count)
                    )
                )
            )
            if out_channels is not None:
                layers.append(
                    nn.Sequential(
                        LayerNorm2d(in_channels, eps=eps),
                        Conv2d(in_channels, out_channels, 2, stride=2),
                    )
                )
        self.features = nn.Sequential(*layers)
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        last_in, last_out, _ = config.block_setting[-1]
        channels = last_in if last_out is None else last_out
        self.classifier = nn.Sequential(
            LayerNorm2d(channels, eps=config.head_norm_eps),
            nn.Flatten(1),
            Linear(channels, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.avgpool(self.features(x)))


def convnext(key: str) -> ConvNeXt:
    if key not in CONVNEXT_CONFIGS:
        known = ", ".join(CONVNEXT_CONFIGS)
        raise KeyError(f"unknown ConvNeXt size {key!r}; known sizes: {known}")
    return ConvNeXt(CONVNEXT_CONFIGS[key])
