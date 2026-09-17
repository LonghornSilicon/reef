"""Hyperparameters for the published ViT sizes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ViTConfig:
    """Architecture hyperparameters for one ViT size."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    patch_size: int = 16
    image_size: int = 224
    num_channels: int = 3
    num_labels: int = 1000
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-12
    qkv_bias: bool = True


VIT_TINY = ViTConfig(
    hidden_size=192,
    num_hidden_layers=12,
    num_attention_heads=3,
    intermediate_size=768,
)

VIT_SMALL = ViTConfig(
    hidden_size=384,
    num_hidden_layers=12,
    num_attention_heads=6,
    intermediate_size=1536,
)

VIT_BASE = ViTConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
)

VIT_LARGE = ViTConfig(
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4096,
)

VIT_HUGE = ViTConfig(
    hidden_size=1280,
    num_hidden_layers=32,
    num_attention_heads=16,
    intermediate_size=5120,
    patch_size=14,
)

VIT_CONFIGS = {
    "ViT-Ti/16": VIT_TINY,
    "ViT-S/16": VIT_SMALL,
    "ViT-B/16": VIT_BASE,
    "ViT-L/16": VIT_LARGE,
    "ViT-H/14": VIT_HUGE,
}

VIT_REPOS = {
    "ViT-Ti/16": "WinKawaks/vit-tiny-patch16-224",
    "ViT-S/16": "WinKawaks/vit-small-patch16-224",
    "ViT-B/16": "google/vit-base-patch16-224",
    "ViT-L/16": "google/vit-large-patch16-224",
    "ViT-H/14": "google/vit-huge-patch14-224-in21k",
}
