"""Hyperparameters for the published TinyStories-Instruct GPT-Neo sizes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GPTNeoConfig:
    """Architecture hyperparameters for one GPT-Neo size."""

    hidden_size: int
    num_layers: int
    num_heads: int
    vocab_size: int = 50257
    max_position_embeddings: int = 2048
    intermediate_size: int | None = None
    activation_function: str = "gelu_new"
    layer_norm_epsilon: float = 1e-5
    window_size: int = 256
    attention_types: tuple[str, ...] = ("global", "local")
    tie_word_embeddings: bool = True

    def attention_type(self, layer: int) -> str:
        return self.attention_types[layer % len(self.attention_types)]


TINYSTORIES_INSTRUCT_1M = GPTNeoConfig(
    hidden_size=64,
    num_layers=8,
    num_heads=16,
)

TINYSTORIES_INSTRUCT_3M = GPTNeoConfig(
    hidden_size=128,
    num_layers=8,
    num_heads=16,
)

TINYSTORIES_INSTRUCT_8M = GPTNeoConfig(
    hidden_size=256,
    num_layers=8,
    num_heads=16,
)

TINYSTORIES_INSTRUCT_28M = GPTNeoConfig(
    hidden_size=512,
    num_layers=8,
    num_heads=16,
)

TINYSTORIES_INSTRUCT_33M = GPTNeoConfig(
    hidden_size=768,
    num_layers=4,
    num_heads=16,
)

GPT_NEO_CONFIGS = {
    "TinyStories-Instruct-1M": TINYSTORIES_INSTRUCT_1M,
    "TinyStories-Instruct-3M": TINYSTORIES_INSTRUCT_3M,
    "TinyStories-Instruct-8M": TINYSTORIES_INSTRUCT_8M,
    "TinyStories-Instruct-28M": TINYSTORIES_INSTRUCT_28M,
    "TinyStories-Instruct-33M": TINYSTORIES_INSTRUCT_33M,
}

GPT_NEO_REPOS = {
    "TinyStories-Instruct-1M": "roneneldan/TinyStories-Instruct-1M",
    "TinyStories-Instruct-3M": "roneneldan/TinyStories-Instruct-3M",
    "TinyStories-Instruct-8M": "roneneldan/TinyStories-Instruct-8M",
    "TinyStories-Instruct-28M": "roneneldan/TinyStories-Instruct-28M",
    "TinyStories-Instruct-33M": "roneneldan/TinyStories-Instruct-33M",
}
