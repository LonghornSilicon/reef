"""Hyperparameters for the published Gemma 3 model sizes."""

from dataclasses import dataclass

ATTENTION_PATTERN = 6


@dataclass(frozen=True)
class Gemma3Config:
    """Text-backbone hyperparameters for one Gemma 3 size."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    sliding_window: int
    query_pre_attn_scalar: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    rope_local_base_freq: float = 10000.0
    rope_global_base_freq: float = 1000000.0
    rope_global_scaling: float = 1.0
    tie_word_embeddings: bool = True
    attention_bias: bool = False

    def is_sliding(self, layer_index: int) -> bool:
        return (layer_index + 1) % ATTENTION_PATTERN != 0


GEMMA3_270M = Gemma3Config(
    hidden_size=640,
    intermediate_size=2048,
    num_hidden_layers=18,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=256,
    vocab_size=262144,
    sliding_window=512,
    query_pre_attn_scalar=256,
    max_position_embeddings=32768,
)

GEMMA3_1B = Gemma3Config(
    hidden_size=1152,
    intermediate_size=6912,
    num_hidden_layers=26,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=256,
    vocab_size=262144,
    sliding_window=512,
    query_pre_attn_scalar=256,
    max_position_embeddings=32768,
)

GEMMA3_4B = Gemma3Config(
    hidden_size=2560,
    intermediate_size=10240,
    num_hidden_layers=34,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=256,
    vocab_size=262208,
    sliding_window=1024,
    query_pre_attn_scalar=256,
    max_position_embeddings=131072,
    rope_global_scaling=8.0,
)

GEMMA3_12B = Gemma3Config(
    hidden_size=3840,
    intermediate_size=15360,
    num_hidden_layers=48,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=256,
    vocab_size=262208,
    sliding_window=1024,
    query_pre_attn_scalar=256,
    max_position_embeddings=131072,
    rope_global_scaling=8.0,
)

# The only size where query_pre_attn_scalar (168) differs from head_dim (128).
GEMMA3_27B = Gemma3Config(
    hidden_size=5376,
    intermediate_size=21504,
    num_hidden_layers=62,
    num_attention_heads=32,
    num_key_value_heads=16,
    head_dim=128,
    vocab_size=262208,
    sliding_window=1024,
    query_pre_attn_scalar=168,
    max_position_embeddings=131072,
    rope_global_scaling=8.0,
)

GEMMA3_CONFIGS = {
    "Gemma3-270M": GEMMA3_270M,
    "Gemma3-1B": GEMMA3_1B,
    "Gemma3-4B": GEMMA3_4B,
    "Gemma3-12B": GEMMA3_12B,
    "Gemma3-27B": GEMMA3_27B,
}

GEMMA3_REPOS = {
    "Gemma3-270M": "google/gemma-3-270m",
    "Gemma3-1B": "google/gemma-3-1b-pt",
    "Gemma3-4B": "google/gemma-3-4b-pt",
    "Gemma3-12B": "google/gemma-3-12b-pt",
    "Gemma3-27B": "google/gemma-3-27b-pt",
}
