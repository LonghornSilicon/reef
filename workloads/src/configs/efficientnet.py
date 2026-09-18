"""Hyperparameters for the published EfficientNet sizes."""

from dataclasses import dataclass

# (expand_ratio, kernel, stride, input_channels, out_channels, num_layers);
# channels are before width_mult and num_layers before depth_mult.
BlockRow = tuple[int, int, int, int, int, int]

MBCONV_SETTING: tuple[BlockRow, ...] = (
    (1, 3, 1, 32, 16, 1),
    (6, 3, 2, 16, 24, 2),
    (6, 5, 2, 24, 40, 2),
    (6, 3, 2, 40, 80, 3),
    (6, 5, 1, 80, 112, 3),
    (6, 5, 2, 112, 192, 4),
    (6, 3, 1, 192, 320, 1),
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

EFFICIENTNET_CONFIGS = {
    "EfficientNet-B0": EFFICIENTNET_B0,
    "EfficientNet-B1": EFFICIENTNET_B1,
    "EfficientNet-B2": EFFICIENTNET_B2,
}

TORCHVISION_BUILDERS = {
    "EfficientNet-B0": "efficientnet_b0",
    "EfficientNet-B1": "efficientnet_b1",
    "EfficientNet-B2": "efficientnet_b2",
}
