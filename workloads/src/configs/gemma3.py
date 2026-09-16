"""Hyperparameters for the published Gemma 3 model sizes.

These are the text backbones. Gemma 3 at 270M and 1B is text-only; at 4B, 12B
and 27B the released checkpoint also carries a SigLIP vision tower, and what is
described here is the language model underneath it.

Two things set Gemma 3 apart from a conventional decoder and both matter to a
hardware model:

* attention alternates between local and global layers. Five of every six
  layers attend only to a sliding window; the sixth attends to the whole
  context. That caps the KV cache far below what the context length suggests.
* the two kinds of layer use different RoPE bases, and on the 4B and larger
  sizes the global layers additionally carry a linear context-extension
  factor.
"""

from dataclasses import dataclass

#: One in every ``ATTENTION_PATTERN`` layers is global; the rest are local.
ATTENTION_PATTERN = 6


@dataclass(frozen=True)
class Gemma3Config:
    """Architecture hyperparameters for one Gemma 3 size.

    Attributes:
        hidden_size: Width of the residual stream.
        intermediate_size: Width of the feed-forward inner projection.
        num_hidden_layers: Number of decoder layers.
        num_attention_heads: Number of query heads.
        num_key_value_heads: Number of key/value heads.
        head_dim: Width of each attention head.
        vocab_size: Number of tokens in the embedding table.
        sliding_window: Context a local layer may attend to.
        query_pre_attn_scalar: The attention logits are scaled by this to the
            power of -1/2. It equals ``head_dim`` at every size except 27B.
        max_position_embeddings: Longest context the checkpoint declares.
        rms_norm_eps: Epsilon of every normalization.
        rope_local_base_freq: RoPE base for the local (sliding) layers.
        rope_global_base_freq: RoPE base for the global (full) layers.
        rope_global_scaling: Linear RoPE extension factor on the global
            layers; ``1.0`` disables it.
        tie_word_embeddings: Whether the LM head reuses the embedding table.
        attention_bias: Whether the attention projections carry a bias.
    """

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
        """Report whether a layer attends locally rather than globally.

        Layers are numbered from zero and every sixth one is global, so the
        global layers sit at indices 5, 11, 17 and so on.

        Args:
            layer_index: Zero-based index of the decoder layer.

        Returns:
            True when the layer uses sliding-window attention.
        """
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

# 27B is the one size whose query_pre_attn_scalar is not head_dim: it is
# hidden_size // num_attention_heads = 168, while head_dim is 128. Deriving
# the attention scale from head_dim here would be wrong.
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

#: Hugging Face repository each config mirrors.
GEMMA3_REPOS = {
    "Gemma3-270M": "google/gemma-3-270m",
    "Gemma3-1B": "google/gemma-3-1b-pt",
    "Gemma3-4B": "google/gemma-3-4b-pt",
    "Gemma3-12B": "google/gemma-3-12b-pt",
    "Gemma3-27B": "google/gemma-3-27b-pt",
}
