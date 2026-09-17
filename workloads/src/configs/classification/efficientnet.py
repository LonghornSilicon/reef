"""Hyperparameters for the published EfficientNet and EfficientNetV2 sizes."""

from dataclasses import dataclass
from typing import Literal

BlockKind = Literal["mbconv", "fused"]

# (expand_ratio, kernel, stride, input_channels, out_channels, num_layers,
#  kind); channels are before width_mult and num_layers before depth_mult.
BlockRow = tuple[int, int, int, int, int, int, BlockKind]

MBCONV_SETTING: tuple[BlockRow, ...] = (
    (1, 3, 1, 32, 16, 1, "mbconv"),
    (6, 3, 2, 16, 24, 2, "mbconv"),
    (6, 5, 2, 24, 40, 2, "mbconv"),
    (6, 3, 2, 40, 80, 3, "mbconv"),
    (6, 5, 1, 80, 112, 3, "mbconv"),
    (6, 5, 2, 112, 192, 4, "mbconv"),
    (6, 3, 1, 192, 320, 1, "mbconv"),
)


@dataclass(frozen=True)
class EfficientNetConfig:
    """Architecture hyperparameters for one EfficientNet size."""

    dropout: float
    inverted_residual_setting: tuple[BlockRow, ...] = MBCONV_SETTING
    width_mult: float = 1.0
    depth_mult: float = 1.0
    # None: 4x the last stage's width-scaled out_channels.
    last_channel: int | None = None
    stochastic_depth_prob: float = 0.2
    num_classes: int = 1000
    norm_eps: float = 1e-5
    norm_momentum: float = 0.1


EFFICIENTNET_B0 = EfficientNetConfig(
    dropout=0.2, width_mult=1.0, depth_mult=1.0
)

EFFICIENTNET_B1 = EfficientNetConfig(
    dropout=0.2, width_mult=1.0, depth_mult=1.1
)

EFFICIENTNET_B2 = EfficientNetConfig(
    dropout=0.3, width_mult=1.1, depth_mult=1.2
)

EFFICIENTNET_B3 = EfficientNetConfig(
    dropout=0.3, width_mult=1.2, depth_mult=1.4
)

EFFICIENTNET_B4 = EfficientNetConfig(
    dropout=0.4, width_mult=1.4, depth_mult=1.8
)

EFFICIENTNET_B5 = EfficientNetConfig(
    dropout=0.4,
    width_mult=1.6,
    depth_mult=2.2,
    norm_eps=1e-3,
    norm_momentum=0.01,
)

EFFICIENTNET_B6 = EfficientNetConfig(
    dropout=0.5,
    width_mult=1.8,
    depth_mult=2.6,
    norm_eps=1e-3,
    norm_momentum=0.01,
)

EFFICIENTNET_B7 = EfficientNetConfig(
    dropout=0.5,
    width_mult=2.0,
    depth_mult=3.1,
    norm_eps=1e-3,
    norm_momentum=0.01,
)

EFFICIENTNET_V2_S = EfficientNetConfig(
    dropout=0.2,
    inverted_residual_setting=(
        (1, 3, 1, 24, 24, 2, "fused"),
        (4, 3, 2, 24, 48, 4, "fused"),
        (4, 3, 2, 48, 64, 4, "fused"),
        (4, 3, 2, 64, 128, 6, "mbconv"),
        (6, 3, 1, 128, 160, 9, "mbconv"),
        (6, 3, 2, 160, 256, 15, "mbconv"),
    ),
    last_channel=1280,
    norm_eps=1e-3,
)

EFFICIENTNET_V2_M = EfficientNetConfig(
    dropout=0.3,
    inverted_residual_setting=(
        (1, 3, 1, 24, 24, 3, "fused"),
        (4, 3, 2, 24, 48, 5, "fused"),
        (4, 3, 2, 48, 80, 5, "fused"),
        (4, 3, 2, 80, 160, 7, "mbconv"),
        (6, 3, 1, 160, 176, 14, "mbconv"),
        (6, 3, 2, 176, 304, 18, "mbconv"),
        (6, 3, 1, 304, 512, 5, "mbconv"),
    ),
    last_channel=1280,
    norm_eps=1e-3,
)

EFFICIENTNET_V2_L = EfficientNetConfig(
    dropout=0.4,
    inverted_residual_setting=(
        (1, 3, 1, 32, 32, 4, "fused"),
        (4, 3, 2, 32, 64, 7, "fused"),
        (4, 3, 2, 64, 96, 7, "fused"),
        (4, 3, 2, 96, 192, 10, "mbconv"),
        (6, 3, 1, 192, 224, 19, "mbconv"),
        (6, 3, 2, 224, 384, 25, "mbconv"),
        (6, 3, 1, 384, 640, 7, "mbconv"),
    ),
    last_channel=1280,
    norm_eps=1e-3,
)

EFFICIENTNET_CONFIGS = {
    "EfficientNet-B0": EFFICIENTNET_B0,
    "EfficientNet-B1": EFFICIENTNET_B1,
    "EfficientNet-B2": EFFICIENTNET_B2,
    "EfficientNet-B3": EFFICIENTNET_B3,
    "EfficientNet-B4": EFFICIENTNET_B4,
    "EfficientNet-B5": EFFICIENTNET_B5,
    "EfficientNet-B6": EFFICIENTNET_B6,
    "EfficientNet-B7": EFFICIENTNET_B7,
    "EfficientNetV2-S": EFFICIENTNET_V2_S,
    "EfficientNetV2-M": EFFICIENTNET_V2_M,
    "EfficientNetV2-L": EFFICIENTNET_V2_L,
}

TORCHVISION_BUILDERS = {
    "EfficientNet-B0": "efficientnet_b0",
    "EfficientNet-B1": "efficientnet_b1",
    "EfficientNet-B2": "efficientnet_b2",
    "EfficientNet-B3": "efficientnet_b3",
    "EfficientNet-B4": "efficientnet_b4",
    "EfficientNet-B5": "efficientnet_b5",
    "EfficientNet-B6": "efficientnet_b6",
    "EfficientNet-B7": "efficientnet_b7",
    "EfficientNetV2-S": "efficientnet_v2_s",
    "EfficientNetV2-M": "efficientnet_v2_m",
    "EfficientNetV2-L": "efficientnet_v2_l",
}
