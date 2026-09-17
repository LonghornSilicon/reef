"""Hyperparameters for the published Swin Transformer sizes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SwinConfig:
    """Architecture hyperparameters for one Swin Transformer size."""

    embed_dim: int
    depths: tuple[int, int, int, int]
    num_heads: tuple[int, int, int, int]
    window_size: int = 7
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    patch_size: int = 4
    image_size: int = 224
    num_labels: int = 1000
    drop_path_rate: float = 0.1
    layer_norm_eps: float = 1e-5
    hidden_act: str = "gelu"
    use_absolute_embeddings: bool = False


SWIN_TINY = SwinConfig(
    embed_dim=96,
    depths=(2, 2, 6, 2),
    num_heads=(3, 6, 12, 24),
)

SWIN_SMALL = SwinConfig(
    embed_dim=96,
    depths=(2, 2, 18, 2),
    num_heads=(3, 6, 12, 24),
)

SWIN_BASE = SwinConfig(
    embed_dim=128,
    depths=(2, 2, 18, 2),
    num_heads=(4, 8, 16, 32),
)

SWIN_LARGE = SwinConfig(
    embed_dim=192,
    depths=(2, 2, 18, 2),
    num_heads=(6, 12, 24, 48),
)

SWIN_CONFIGS = {
    "Swin-T": SWIN_TINY,
    "Swin-S": SWIN_SMALL,
    "Swin-B": SWIN_BASE,
    "Swin-L": SWIN_LARGE,
}

SWIN_REPOS = {
    "Swin-T": "microsoft/swin-tiny-patch4-window7-224",
    "Swin-S": "microsoft/swin-small-patch4-window7-224",
    "Swin-B": "microsoft/swin-base-patch4-window7-224",
    "Swin-L": "microsoft/swin-large-patch4-window7-224",
}
