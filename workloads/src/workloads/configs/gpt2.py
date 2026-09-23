"""Hyperparameters for the published GPT-2 sizes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GPT2Config:
    """Architecture hyperparameters for one GPT-2 size."""

    n_embd: int
    n_layer: int
    n_head: int
    vocab_size: int = 50257
    n_positions: int = 1024
    n_inner: int | None = None
    activation_function: str = "gelu_new"
    layer_norm_epsilon: float = 1e-5
    scale_attn_weights: bool = True
    tie_word_embeddings: bool = True


GPT2_SMALL = GPT2Config(
    n_embd=768,
    n_layer=12,
    n_head=12,
)

GPT2_CONFIGS = {
    "GPT-2": GPT2_SMALL,
}

GPT2_REPOS = {
    "GPT-2": "openai-community/gpt2",
}
