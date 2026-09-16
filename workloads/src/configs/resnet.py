"""Hyperparameters for the published ResNet depths."""

from dataclasses import dataclass
from typing import Literal

BlockKind = Literal["basic", "bottleneck"]

#: Channel width of each of the four stages, before block expansion.
STAGE_PLANES = (64, 128, 256, 512)

#: Output channels of a block divided by its ``planes``.
EXPANSION: dict[BlockKind, int] = {"basic": 1, "bottleneck": 4}


@dataclass(frozen=True)
class ResNetConfig:
    """Architecture hyperparameters for one ResNet depth."""

    block: BlockKind
    blocks_per_stage: tuple[int, int, int, int]
    num_classes: int = 1000
    stem_channels: int = 64
    norm_eps: float = 1e-5

    @property
    def expansion(self) -> int:
        return EXPANSION[self.block]

    @property
    def feature_channels(self) -> int:
        return STAGE_PLANES[-1] * self.expansion


RESNET_18 = ResNetConfig(block="basic", blocks_per_stage=(2, 2, 2, 2))

RESNET_34 = ResNetConfig(block="basic", blocks_per_stage=(3, 4, 6, 3))

RESNET_50 = ResNetConfig(block="bottleneck", blocks_per_stage=(3, 4, 6, 3))

RESNET_101 = ResNetConfig(block="bottleneck", blocks_per_stage=(3, 4, 23, 3))

RESNET_CONFIGS = {
    "ResNet-18": RESNET_18,
    "ResNet-34": RESNET_34,
    "ResNet-50": RESNET_50,
    "ResNet-101": RESNET_101,
}

TORCHVISION_BUILDERS = {
    "ResNet-18": "resnet18",
    "ResNet-34": "resnet34",
    "ResNet-50": "resnet50",
    "ResNet-101": "resnet101",
}
