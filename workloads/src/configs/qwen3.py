"""Hyperparameters for the published Qwen3 dense model sizes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen3Config:
    """Architecture hyperparameters for one Qwen3 model size."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    tie_word_embeddings: bool
    vocab_size: int = 151936
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 40960
    attention_bias: bool = False


QWEN3_0_6B = Qwen3Config(
    hidden_size=1024,
    intermediate_size=3072,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    tie_word_embeddings=True,
)

QWEN3_1_7B = Qwen3Config(
    hidden_size=2048,
    intermediate_size=6144,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    tie_word_embeddings=True,
)

QWEN3_4B = Qwen3Config(
    hidden_size=2560,
    intermediate_size=9728,
    num_hidden_layers=36,
    num_attention_heads=32,
    num_key_value_heads=8,
    tie_word_embeddings=True,
)

QWEN3_8B = Qwen3Config(
    hidden_size=4096,
    intermediate_size=12288,
    num_hidden_layers=36,
    num_attention_heads=32,
    num_key_value_heads=8,
    tie_word_embeddings=False,
)

QWEN3_14B = Qwen3Config(
    hidden_size=5120,
    intermediate_size=17408,
    num_hidden_layers=40,
    num_attention_heads=40,
    num_key_value_heads=8,
    tie_word_embeddings=False,
)

QWEN3_32B = Qwen3Config(
    hidden_size=5120,
    intermediate_size=25600,
    num_hidden_layers=64,
    num_attention_heads=64,
    num_key_value_heads=8,
    tie_word_embeddings=False,
)

QWEN3_CONFIGS = {
    "Qwen3-0.6B": QWEN3_0_6B,
    "Qwen3-1.7B": QWEN3_1_7B,
    "Qwen3-4B": QWEN3_4B,
    "Qwen3-8B": QWEN3_8B,
    "Qwen3-14B": QWEN3_14B,
    "Qwen3-32B": QWEN3_32B,
}
