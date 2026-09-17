"""Hyperparameters for the published VGG depths."""

from dataclasses import dataclass

# Conv output channels in order; "M" marks a 2x2 stride-2 max-pool.
LayerSpec = tuple[int | str, ...]

# fmt: off
LAYERS_11: LayerSpec = (
    64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M",
)

LAYERS_13: LayerSpec = (
    64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M",
)

LAYERS_16: LayerSpec = (
    64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
    512, 512, 512, "M", 512, 512, 512, "M",
)

LAYERS_19: LayerSpec = (
    64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
    512, 512, 512, 512, "M", 512, 512, 512, 512, "M",
)
# fmt: on


@dataclass(frozen=True)
class VGGConfig:
    """Architecture hyperparameters for one VGG depth."""

    layers: LayerSpec
    batch_norm: bool
    num_classes: int = 1000
    dropout: float = 0.5


VGG_11 = VGGConfig(layers=LAYERS_11, batch_norm=False)

VGG_11_BN = VGGConfig(layers=LAYERS_11, batch_norm=True)

VGG_13 = VGGConfig(layers=LAYERS_13, batch_norm=False)

VGG_13_BN = VGGConfig(layers=LAYERS_13, batch_norm=True)

VGG_16 = VGGConfig(layers=LAYERS_16, batch_norm=False)

VGG_16_BN = VGGConfig(layers=LAYERS_16, batch_norm=True)

VGG_19 = VGGConfig(layers=LAYERS_19, batch_norm=False)

VGG_19_BN = VGGConfig(layers=LAYERS_19, batch_norm=True)

VGG_CONFIGS = {
    "VGG-11": VGG_11,
    "VGG-11-BN": VGG_11_BN,
    "VGG-13": VGG_13,
    "VGG-13-BN": VGG_13_BN,
    "VGG-16": VGG_16,
    "VGG-16-BN": VGG_16_BN,
    "VGG-19": VGG_19,
    "VGG-19-BN": VGG_19_BN,
}

TORCHVISION_BUILDERS = {
    "VGG-11": "vgg11",
    "VGG-11-BN": "vgg11_bn",
    "VGG-13": "vgg13",
    "VGG-13-BN": "vgg13_bn",
    "VGG-16": "vgg16",
    "VGG-16-BN": "vgg16_bn",
    "VGG-19": "vgg19",
    "VGG-19-BN": "vgg19_bn",
}
