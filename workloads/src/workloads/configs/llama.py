"""Hyperparameters for the published Llama-architecture sizes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LlamaConfig:
    """Architecture hyperparameters for one Llama-architecture model."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    head_dim: int = 64
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    attention_bias: bool = False
    mlp_bias: bool = False


SMOLLM2_135M = LlamaConfig(
    vocab_size=49152,
    hidden_size=576,
    intermediate_size=1536,
    num_hidden_layers=30,
    num_attention_heads=9,
    num_key_value_heads=3,
    rope_theta=100000.0,
    max_position_embeddings=8192,
    tie_word_embeddings=True,
)

LLAMA_CONFIGS = {
    "SmolLM2-135M": SMOLLM2_135M,
}

LLAMA_REPOS = {
    "SmolLM2-135M": "HuggingFaceTB/SmolLM2-135M",
}
