"""Hyperparameters for the published BLOOM sizes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BloomConfig:
    """Architecture hyperparameters for one BLOOM model size."""

    hidden_size: int
    n_layer: int = 24
    n_head: int = 16
    vocab_size: int = 250880
    layer_norm_epsilon: float = 1e-5
    apply_residual_connection_post_layernorm: bool = False
    tie_word_embeddings: bool = True


BLOOM_560M = BloomConfig(
    hidden_size=1024,
)

BLOOM_1B1 = BloomConfig(
    hidden_size=1536,
)

BLOOM_CONFIGS = {
    "BLOOM-560M": BLOOM_560M,
    "BLOOM-1B1": BLOOM_1B1,
}

BLOOM_REPOS = {
    "BLOOM-560M": "bigscience/bloom-560m",
    "BLOOM-1B1": "bigscience/bloom-1b1",
}
