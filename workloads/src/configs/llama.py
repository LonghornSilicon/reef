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
    # Llama 3 frequency-band rescaling: factor, low_freq_factor,
    # high_freq_factor, original_max_position_embeddings.
    rope_scaling: dict[str, float] | None = None


TINYLLAMA_1_1B = LlamaConfig(
    vocab_size=32000,
    hidden_size=2048,
    intermediate_size=5632,
    num_hidden_layers=22,
    num_attention_heads=32,
    num_key_value_heads=4,
    rope_theta=10000.0,
    max_position_embeddings=2048,
    tie_word_embeddings=False,
)

LLAMA_3_2_1B = LlamaConfig(
    vocab_size=128256,
    hidden_size=2048,
    intermediate_size=8192,
    num_hidden_layers=16,
    num_attention_heads=32,
    num_key_value_heads=8,
    rope_theta=500000.0,
    max_position_embeddings=131072,
    tie_word_embeddings=True,
    rope_scaling={
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    },
)

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

SMOLLM2_360M = LlamaConfig(
    vocab_size=49152,
    hidden_size=960,
    intermediate_size=2560,
    num_hidden_layers=32,
    num_attention_heads=15,
    num_key_value_heads=5,
    rope_theta=100000.0,
    max_position_embeddings=8192,
    tie_word_embeddings=True,
)

LLAMA_CONFIGS = {
    "TinyLlama-1.1B": TINYLLAMA_1_1B,
    "Llama-3.2-1B": LLAMA_3_2_1B,
    "SmolLM2-135M": SMOLLM2_135M,
    "SmolLM2-360M": SMOLLM2_360M,
}

LLAMA_REPOS = {
    "TinyLlama-1.1B": "TinyLlama/TinyLlama_v1.1",
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    "SmolLM2-135M": "HuggingFaceTB/SmolLM2-135M",
    "SmolLM2-360M": "HuggingFaceTB/SmolLM2-360M",
}

# meta-llama/Llama-3.2-1B is gated; the ungated unsloth re-upload has the
# same config and weights.
LLAMA_MIRRORS = {
    "Llama-3.2-1B": "unsloth/Llama-3.2-1B",
}
