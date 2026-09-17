"""Hyperparameters for the published ConvNeXt sizes."""

from dataclasses import dataclass

# (input_channels, out_channels, num_layers); out_channels None means no
# downsampling layer follows the stage.
BlockRow = tuple[int, int | None, int]


@dataclass(frozen=True)
class ConvNeXtConfig:
    """Architecture hyperparameters for one ConvNeXt size."""

    block_setting: tuple[BlockRow, ...]
    stochastic_depth_prob: float
    layer_scale: float = 1e-6
    num_classes: int = 1000
    norm_eps: float = 1e-6
    head_norm_eps: float = 1e-6


CONVNEXT_TINY = ConvNeXtConfig(
    block_setting=((96, 192, 3), (192, 384, 3), (384, 768, 9), (768, None, 3)),
    stochastic_depth_prob=0.1,
)

CONVNEXT_SMALL = ConvNeXtConfig(
    block_setting=(
        (96, 192, 3),
        (192, 384, 3),
        (384, 768, 27),
        (768, None, 3),
    ),
    stochastic_depth_prob=0.4,
)

CONVNEXT_BASE = ConvNeXtConfig(
    block_setting=(
        (128, 256, 3),
        (256, 512, 3),
        (512, 1024, 27),
        (1024, None, 3),
    ),
    stochastic_depth_prob=0.5,
)

CONVNEXT_LARGE = ConvNeXtConfig(
    block_setting=(
        (192, 384, 3),
        (384, 768, 3),
        (768, 1536, 27),
        (1536, None, 3),
    ),
    stochastic_depth_prob=0.5,
)

# transformers hardcodes eps=1e-6 in every LayerNorm except the one before
# the classifier, which reads config.layer_norm_eps (1e-12 on the hub).
CONVNEXT_XLARGE = ConvNeXtConfig(
    block_setting=(
        (256, 512, 3),
        (512, 1024, 3),
        (1024, 2048, 27),
        (2048, None, 3),
    ),
    stochastic_depth_prob=0.0,
    num_classes=21841,
    head_norm_eps=1e-12,
)

CONVNEXT_CONFIGS = {
    "ConvNeXt-T": CONVNEXT_TINY,
    "ConvNeXt-S": CONVNEXT_SMALL,
    "ConvNeXt-B": CONVNEXT_BASE,
    "ConvNeXt-L": CONVNEXT_LARGE,
    "ConvNeXt-XL": CONVNEXT_XLARGE,
}

TORCHVISION_BUILDERS = {
    "ConvNeXt-T": "convnext_tiny",
    "ConvNeXt-S": "convnext_small",
    "ConvNeXt-B": "convnext_base",
    "ConvNeXt-L": "convnext_large",
}

CONVNEXT_REPOS = {
    "ConvNeXt-XL": "facebook/convnext-xlarge-224-22k",
}
