"""Hyperparameters for the published CLIP sizes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CLIPVisionConfig:
    """Vision-tower hyperparameters for one CLIP size."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    patch_size: int
    image_size: int = 224
    num_channels: int = 3
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5


@dataclass(frozen=True)
class CLIPTextConfig:
    """Text-tower hyperparameters for one CLIP size."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int = 77
    vocab_size: int = 49408
    eos_token_id: int = 49407
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5


@dataclass(frozen=True)
class CLIPConfig:
    """Two-tower hyperparameters for one CLIP size."""

    vision: CLIPVisionConfig
    text: CLIPTextConfig
    projection_dim: int
    logit_scale_init_value: float = 2.6592


CLIP_B32 = CLIPConfig(
    vision=CLIPVisionConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        patch_size=32,
    ),
    text=CLIPTextConfig(
        hidden_size=512,
        num_hidden_layers=12,
        num_attention_heads=8,
        intermediate_size=2048,
    ),
    projection_dim=512,
)

CLIP_B16 = CLIPConfig(
    vision=CLIPVisionConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        patch_size=16,
    ),
    text=CLIPTextConfig(
        hidden_size=512,
        num_hidden_layers=12,
        num_attention_heads=8,
        intermediate_size=2048,
    ),
    projection_dim=512,
)

CLIP_L14 = CLIPConfig(
    vision=CLIPVisionConfig(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        patch_size=14,
    ),
    text=CLIPTextConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
    ),
    projection_dim=768,
)

CLIP_CONFIGS = {
    "CLIP-ViT-B/32": CLIP_B32,
    "CLIP-ViT-B/16": CLIP_B16,
    "CLIP-ViT-L/14": CLIP_L14,
}

CLIP_REPOS = {
    "CLIP-ViT-B/32": "openai/clip-vit-base-patch32",
    "CLIP-ViT-B/16": "openai/clip-vit-base-patch16",
    "CLIP-ViT-L/14": "openai/clip-vit-large-patch14",
}
