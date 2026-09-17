"""Hyperparameters for the published MobileViT sizes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MobileViTConfig:
    """Architecture hyperparameters for one MobileViT size."""

    hidden_sizes: tuple[int, int, int]
    neck_hidden_sizes: tuple[int, int, int, int, int, int, int]
    expand_ratio: float
    num_attention_heads: int = 4
    mlp_ratio: float = 2.0
    patch_size: int = 2
    image_size: int = 256
    num_labels: int = 1000
    hidden_act: str = "silu"
    conv_kernel_size: int = 3
    output_stride: int = 32
    layer_norm_eps: float = 1e-5


MOBILEVIT_XXS = MobileViTConfig(
    hidden_sizes=(64, 80, 96),
    neck_hidden_sizes=(16, 16, 24, 48, 64, 80, 320),
    expand_ratio=2.0,
)

MOBILEVIT_XS = MobileViTConfig(
    hidden_sizes=(96, 120, 144),
    neck_hidden_sizes=(16, 32, 48, 64, 80, 96, 384),
    expand_ratio=4.0,
)

MOBILEVIT_S = MobileViTConfig(
    hidden_sizes=(144, 192, 240),
    neck_hidden_sizes=(16, 32, 64, 96, 128, 160, 640),
    expand_ratio=4.0,
)

MOBILEVIT_CONFIGS = {
    "MobileViT-XXS": MOBILEVIT_XXS,
    "MobileViT-XS": MOBILEVIT_XS,
    "MobileViT-S": MOBILEVIT_S,
}

MOBILEVIT_REPOS = {
    "MobileViT-XXS": "apple/mobilevit-xx-small",
    "MobileViT-XS": "apple/mobilevit-x-small",
    "MobileViT-S": "apple/mobilevit-small",
}
