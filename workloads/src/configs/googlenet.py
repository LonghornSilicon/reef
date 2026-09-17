"""Hyperparameters for the published GoogLeNet model."""

from dataclasses import dataclass

# (in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj) for the
# blocks 3a, 3b, 4a-4e, 5a, 5b; a max-pool follows 3b and 4e.
InceptionSpec = tuple[int, int, int, int, int, int, int]

INCEPTION_BLOCKS: tuple[InceptionSpec, ...] = (
    (192, 64, 96, 128, 16, 32, 32),
    (256, 128, 128, 192, 32, 96, 64),
    (480, 192, 96, 208, 16, 48, 64),
    (512, 160, 112, 224, 24, 64, 64),
    (512, 128, 128, 256, 24, 64, 64),
    (512, 112, 144, 288, 32, 64, 64),
    (528, 256, 160, 320, 32, 128, 128),
    (832, 256, 160, 320, 32, 128, 128),
    (832, 384, 192, 384, 48, 128, 128),
)


@dataclass(frozen=True)
class GoogLeNetConfig:
    """Architecture hyperparameters for GoogLeNet."""

    inception_blocks: tuple[InceptionSpec, ...]
    aux_logits: bool = True
    num_classes: int = 1000
    dropout: float = 0.2
    dropout_aux: float = 0.7
    norm_eps: float = 1e-3


GOOGLENET = GoogLeNetConfig(inception_blocks=INCEPTION_BLOCKS)

GOOGLENET_CONFIGS = {
    "GoogLeNet": GOOGLENET,
}

TORCHVISION_BUILDERS = {
    "GoogLeNet": "googlenet",
}
