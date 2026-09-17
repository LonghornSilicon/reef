"""Hyperparameters for the published DINOv2 sizes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Dinov2Config:
    """Architecture hyperparameters for one DINOv2 size."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    use_swiglu_ffn: bool
    mlp_ratio: int = 4
    hidden_act: str = "gelu"
    layerscale_value: float = 1.0
    patch_size: int = 14
    image_size: int = 518
    num_channels: int = 3
    num_labels: int = 1000
    layer_norm_eps: float = 1e-6
    qkv_bias: bool = True


DINOV2_SMALL = Dinov2Config(
    hidden_size=384,
    num_hidden_layers=12,
    num_attention_heads=6,
    use_swiglu_ffn=False,
)

DINOV2_BASE = Dinov2Config(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    use_swiglu_ffn=False,
)

DINOV2_LARGE = Dinov2Config(
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    use_swiglu_ffn=False,
)

DINOV2_GIANT = Dinov2Config(
    hidden_size=1536,
    num_hidden_layers=40,
    num_attention_heads=24,
    use_swiglu_ffn=True,
)

DINOV2_CONFIGS = {
    "DINOv2-S": DINOV2_SMALL,
    "DINOv2-B": DINOV2_BASE,
    "DINOv2-L": DINOV2_LARGE,
    "DINOv2-g": DINOV2_GIANT,
}

DINOV2_REPOS = {
    "DINOv2-S": "facebook/dinov2-small",
    "DINOv2-B": "facebook/dinov2-base",
    "DINOv2-L": "facebook/dinov2-large",
    "DINOv2-g": "facebook/dinov2-giant",
}
