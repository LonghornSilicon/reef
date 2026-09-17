"""Hyperparameters for the published Qwen2.5 dense model sizes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen2Config:
    """Architecture hyperparameters for one Qwen2.5 model size."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    max_position_embeddings: int
    num_key_value_heads: int = 2
    vocab_size: int = 151936
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    # HF Qwen2 has no attention_bias knob: q/k/v projections always carry a
    # bias and o_proj never does.
    qkv_bias: bool = True
    use_sliding_window: bool = False
    sliding_window: int | None = None
    tie_word_embeddings: bool = True


QWEN2_5_0_5B = Qwen2Config(
    hidden_size=896,
    intermediate_size=4864,
    num_hidden_layers=24,
    num_attention_heads=14,
    max_position_embeddings=32768,
)

QWEN2_5_1_5B = Qwen2Config(
    hidden_size=1536,
    intermediate_size=8960,
    num_hidden_layers=28,
    num_attention_heads=12,
    max_position_embeddings=131072,
)

QWEN2_CONFIGS = {
    "Qwen2.5-0.5B": QWEN2_5_0_5B,
    "Qwen2.5-1.5B": QWEN2_5_1_5B,
}

QWEN2_REPOS = {
    "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
}
