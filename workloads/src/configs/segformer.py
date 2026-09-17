"""Hyperparameters for the published SegFormer sizes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SegformerConfig:
    """Architecture hyperparameters for one SegFormer size."""

    hidden_sizes: tuple[int, int, int, int]
    depths: tuple[int, int, int, int]
    decoder_hidden_size: int
    num_attention_heads: tuple[int, int, int, int] = (1, 2, 5, 8)
    sr_ratios: tuple[int, int, int, int] = (8, 4, 2, 1)
    patch_sizes: tuple[int, int, int, int] = (7, 3, 3, 3)
    strides: tuple[int, int, int, int] = (4, 2, 2, 2)
    mlp_ratios: tuple[int, int, int, int] = (4, 4, 4, 4)
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    num_labels: int = 1000
    reshape_last_stage: bool = True


SEGFORMER_B0 = SegformerConfig(
    hidden_sizes=(32, 64, 160, 256),
    depths=(2, 2, 2, 2),
    decoder_hidden_size=256,
)

SEGFORMER_B1 = SegformerConfig(
    hidden_sizes=(64, 128, 320, 512),
    depths=(2, 2, 2, 2),
    decoder_hidden_size=256,
)

SEGFORMER_B2 = SegformerConfig(
    hidden_sizes=(64, 128, 320, 512),
    depths=(3, 4, 6, 3),
    decoder_hidden_size=768,
)

SEGFORMER_B3 = SegformerConfig(
    hidden_sizes=(64, 128, 320, 512),
    depths=(3, 4, 18, 3),
    decoder_hidden_size=768,
)

SEGFORMER_B4 = SegformerConfig(
    hidden_sizes=(64, 128, 320, 512),
    depths=(3, 8, 27, 3),
    decoder_hidden_size=768,
)

SEGFORMER_B5 = SegformerConfig(
    hidden_sizes=(64, 128, 320, 512),
    depths=(3, 6, 40, 3),
    decoder_hidden_size=768,
)

SEGFORMER_CONFIGS = {
    "SegFormer-B0": SEGFORMER_B0,
    "SegFormer-B1": SEGFORMER_B1,
    "SegFormer-B2": SEGFORMER_B2,
    "SegFormer-B3": SEGFORMER_B3,
    "SegFormer-B4": SEGFORMER_B4,
    "SegFormer-B5": SEGFORMER_B5,
}

SEGFORMER_REPOS = {
    "SegFormer-B0": "nvidia/mit-b0",
    "SegFormer-B1": "nvidia/mit-b1",
    "SegFormer-B2": "nvidia/mit-b2",
    "SegFormer-B3": "nvidia/mit-b3",
    "SegFormer-B4": "nvidia/mit-b4",
    "SegFormer-B5": "nvidia/mit-b5",
}
