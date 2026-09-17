"""Hyperparameters for the published MobileNetV3 sizes."""

from dataclasses import dataclass
from typing import Literal

Activation = Literal["RE", "HS"]

# (input_channels, kernel, expanded_channels, out_channels, use_se,
#  activation, stride, dilation)
BneckRow = tuple[int, int, int, int, bool, Activation, int, int]


@dataclass(frozen=True)
class MobileNetV3Config:
    """Architecture hyperparameters for one MobileNetV3 size."""

    inverted_residual_setting: tuple[BneckRow, ...]
    last_channel: int
    num_classes: int = 1000
    dropout: float = 0.2
    norm_eps: float = 1e-3
    norm_momentum: float = 0.01


MOBILENET_V3_SMALL = MobileNetV3Config(
    inverted_residual_setting=(
        (16, 3, 16, 16, True, "RE", 2, 1),
        (16, 3, 72, 24, False, "RE", 2, 1),
        (24, 3, 88, 24, False, "RE", 1, 1),
        (24, 5, 96, 40, True, "HS", 2, 1),
        (40, 5, 240, 40, True, "HS", 1, 1),
        (40, 5, 240, 40, True, "HS", 1, 1),
        (40, 5, 120, 48, True, "HS", 1, 1),
        (48, 5, 144, 48, True, "HS", 1, 1),
        (48, 5, 288, 96, True, "HS", 2, 1),
        (96, 5, 576, 96, True, "HS", 1, 1),
        (96, 5, 576, 96, True, "HS", 1, 1),
    ),
    last_channel=1024,
)

MOBILENET_V3_LARGE = MobileNetV3Config(
    inverted_residual_setting=(
        (16, 3, 16, 16, False, "RE", 1, 1),
        (16, 3, 64, 24, False, "RE", 2, 1),
        (24, 3, 72, 24, False, "RE", 1, 1),
        (24, 5, 72, 40, True, "RE", 2, 1),
        (40, 5, 120, 40, True, "RE", 1, 1),
        (40, 5, 120, 40, True, "RE", 1, 1),
        (40, 3, 240, 80, False, "HS", 2, 1),
        (80, 3, 200, 80, False, "HS", 1, 1),
        (80, 3, 184, 80, False, "HS", 1, 1),
        (80, 3, 184, 80, False, "HS", 1, 1),
        (80, 3, 480, 112, True, "HS", 1, 1),
        (112, 3, 672, 112, True, "HS", 1, 1),
        (112, 5, 672, 160, True, "HS", 2, 1),
        (160, 5, 960, 160, True, "HS", 1, 1),
        (160, 5, 960, 160, True, "HS", 1, 1),
    ),
    last_channel=1280,
)

MOBILENET_V3_CONFIGS = {
    "MobileNetV3-Small": MOBILENET_V3_SMALL,
    "MobileNetV3-Large": MOBILENET_V3_LARGE,
}

TORCHVISION_BUILDERS = {
    "MobileNetV3-Small": "mobilenet_v3_small",
    "MobileNetV3-Large": "mobilenet_v3_large",
}
