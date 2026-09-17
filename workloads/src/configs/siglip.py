"""Hyperparameters for the published SigLIP sizes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SiglipVisionConfig:
    """Vision-tower hyperparameters for one SigLIP size."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    patch_size: int
    image_size: int
    num_channels: int = 3
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6


@dataclass(frozen=True)
class SiglipTextConfig:
    """Text-tower hyperparameters for one SigLIP size."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    projection_size: int
    max_position_embeddings: int = 64
    vocab_size: int = 32000
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6


@dataclass(frozen=True)
class SiglipConfig:
    """Two-tower hyperparameters for one SigLIP size."""

    vision: SiglipVisionConfig
    text: SiglipTextConfig


SIGLIP_B16 = SiglipConfig(
    vision=SiglipVisionConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        patch_size=16,
        image_size=224,
    ),
    text=SiglipTextConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        projection_size=768,
    ),
)

SIGLIP_L16 = SiglipConfig(
    vision=SiglipVisionConfig(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        patch_size=16,
        image_size=256,
    ),
    text=SiglipTextConfig(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        projection_size=1024,
    ),
)

SIGLIP_SO400M = SiglipConfig(
    vision=SiglipVisionConfig(
        hidden_size=1152,
        num_hidden_layers=27,
        num_attention_heads=16,
        intermediate_size=4304,
        patch_size=14,
        image_size=384,
    ),
    text=SiglipTextConfig(
        hidden_size=1152,
        num_hidden_layers=27,
        num_attention_heads=16,
        intermediate_size=4304,
        projection_size=1152,
    ),
)

SIGLIP_CONFIGS = {
    "SigLIP-B/16": SIGLIP_B16,
    "SigLIP-L/16": SIGLIP_L16,
    "SigLIP-So400m/14": SIGLIP_SO400M,
}

SIGLIP_REPOS = {
    "SigLIP-B/16": "google/siglip-base-patch16-224",
    "SigLIP-L/16": "google/siglip-large-patch16-256",
    "SigLIP-So400m/14": "google/siglip-so400m-patch14-384",
}
