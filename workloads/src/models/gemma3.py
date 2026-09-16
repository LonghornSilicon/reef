"""Gemma 3 causal language model built from the operator library.

Module and parameter names follow ``transformers``' ``Gemma3ForCausalLM``, so
a published text-backbone ``state_dict`` loads with ``strict=True``.

The structural differences from :mod:`models.qwen3`, all of which change the
arithmetic rather than just the naming:

* every decoder layer has four normalizations, not two -- the feed-forward
  block is wrapped on both sides as well as the attention block;
* embeddings are scaled by ``sqrt(hidden_size)`` on the way in;
* attention alternates local and global layers, each with its own RoPE table;
* the attention logits are scaled by ``query_pre_attn_scalar ** -0.5`` rather
  than by the head width.
"""

import torch
from torch import nn

from configs.gemma3 import GEMMA3_CONFIGS, Gemma3Config
from operators.activation import GELU
from operators.attention import GroupedQueryAttention
from operators.embedding import Embedding
from operators.linear import Linear
from operators.normalization import GemmaRMSNorm
from operators.rotary import RotaryEmbedding

LayerCache = tuple[torch.Tensor, torch.Tensor]


class Gemma3Attention(nn.Module):
    """Self-attention with per-head Q/K normalization and RoPE.

    A layer is either local or global. A local layer attends to a sliding
    window and uses the shorter RoPE base; a global layer attends to the whole
    context and uses the longer one, optionally linearly extended.
    """

    def __init__(
        self,
        config: Gemma3Config,
        rotary: RotaryEmbedding,
        sliding: bool,
    ) -> None:
        """Build the projections, head norms and attention kernel.

        Args:
            config: Model hyperparameters.
            rotary: Rotary embedding for this layer's attention kind.
            sliding: Whether this layer attends to a sliding window.
        """
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.sliding = sliding
        query_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        bias = config.attention_bias
        self.q_proj = Linear(config.hidden_size, query_dim, bias=bias)
        self.k_proj = Linear(config.hidden_size, kv_dim, bias=bias)
        self.v_proj = Linear(config.hidden_size, kv_dim, bias=bias)
        self.o_proj = Linear(query_dim, config.hidden_size, bias=bias)
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary = rotary
        self.attention = GroupedQueryAttention(
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            scale=config.query_pre_attn_scalar**-0.5,
            sliding_window=config.sliding_window if sliding else None,
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past: LayerCache | None,
    ) -> tuple[torch.Tensor, LayerCache]:
        """Attend over the sequence, extending any cached keys and values.

        Args:
            x: Normalized hidden states shaped ``(batch, length, hidden)``.
            cos: Cosine table shaped ``(length, head_dim)``.
            sin: Sine table shaped ``(length, head_dim)``.
            past: Keys and values from earlier steps, or ``None``.

        Returns:
            The attention output and the updated key/value cache.
        """
        batch, length, _ = x.shape
        query = self.q_proj(x).reshape(
            batch, length, self.num_heads, self.head_dim
        )
        key = self.k_proj(x).reshape(
            batch, length, self.num_kv_heads, self.head_dim
        )
        value = self.v_proj(x).reshape(
            batch, length, self.num_kv_heads, self.head_dim
        )
        # Q/K norm acts per head over head_dim, before the rotation.
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        query = self.rotary.apply_rotary(query, cos, sin)
        key = self.rotary.apply_rotary(key, cos, sin)
        if past is not None:
            past_key, past_value = past
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
        out = self.attention(query, key, value)
        out = out.transpose(1, 2).reshape(
            batch, length, self.num_heads * self.head_dim
        )
        return self.o_proj(out), (key, value)


class Gemma3MLP(nn.Module):
    """Gated feed-forward network with a GELU gate."""

    def __init__(self, config: Gemma3Config) -> None:
        """Build the gate, up and down projections.

        Args:
            config: Model hyperparameters.
        """
        super().__init__()
        hidden, inner = config.hidden_size, config.intermediate_size
        self.gate_proj = Linear(hidden, inner, bias=False)
        self.up_proj = Linear(hidden, inner, bias=False)
        self.down_proj = Linear(inner, hidden, bias=False)
        self.act_fn = GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated feed-forward transform.

        Args:
            x: Tensor shaped ``(batch, length, hidden)``.

        Returns:
            Tensor of the same shape as ``x``.
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Gemma3DecoderLayer(nn.Module):
    """Transformer block with both blocks normalized on entry and exit."""

    def __init__(
        self,
        config: Gemma3Config,
        rotary: RotaryEmbedding,
        sliding: bool,
    ) -> None:
        """Build the attention block, MLP and the four norms.

        Args:
            config: Model hyperparameters.
            rotary: Rotary embedding for this layer's attention kind.
            sliding: Whether this layer attends to a sliding window.
        """
        super().__init__()
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.self_attn = Gemma3Attention(config, rotary, sliding)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = GemmaRMSNorm(hidden, eps=eps)
        self.post_attention_layernorm = GemmaRMSNorm(hidden, eps=eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(hidden, eps=eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(hidden, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past: LayerCache | None,
    ) -> tuple[torch.Tensor, LayerCache]:
        """Run the block over ``x``.

        Args:
            x: Hidden states shaped ``(batch, length, hidden)``.
            cos: Cosine table shaped ``(length, head_dim)``.
            sin: Sine table shaped ``(length, head_dim)``.
            past: Keys and values from earlier steps, or ``None``.

        Returns:
            The updated hidden states and this layer's key/value cache.
        """
        attended, present = self.self_attn(
            self.input_layernorm(x), cos, sin, past
        )
        x = x + self.post_attention_layernorm(attended)
        fed = self.mlp(self.pre_feedforward_layernorm(x))
        x = x + self.post_feedforward_layernorm(fed)
        return x, present


class Gemma3TextModel(nn.Module):
    """Embedding table, decoder stack and final norm."""

    def __init__(self, config: Gemma3Config) -> None:
        """Build the decoder stack and both rotary tables.

        Args:
            config: Model hyperparameters.
        """
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        # The two attention kinds use different RoPE bases, so both tables are
        # built once here and shared by the layers that need them.
        self.rotary_local = RotaryEmbedding(
            config.head_dim, config.rope_local_base_freq
        )
        self.rotary_global = RotaryEmbedding(
            config.head_dim,
            config.rope_global_base_freq,
            scaling_factor=config.rope_global_scaling,
        )
        self.layers = nn.ModuleList(
            Gemma3DecoderLayer(
                config,
                self.rotary_local
                if config.is_sliding(index)
                else self.rotary_global,
                sliding=config.is_sliding(index),
            )
            for index in range(config.num_hidden_layers)
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: list[LayerCache] | None = None,
    ) -> tuple[torch.Tensor, list[LayerCache]]:
        """Embed ``input_ids`` and run the decoder stack.

        Args:
            input_ids: Integer tensor shaped ``(batch, length)``.
            past_key_values: Per-layer caches from earlier steps, or ``None``.

        Returns:
            Final hidden states and the per-layer key/value caches.
        """
        length = input_ids.shape[1]
        past_length = (
            0 if past_key_values is None else past_key_values[0][0].shape[2]
        )
        x = self.embed_tokens(input_ids)
        # Gemma scales the embeddings on the way in. The multiplier is
        # rounded to the hidden dtype first, matching the reference.
        scale = torch.tensor(
            self.config.hidden_size**0.5, dtype=x.dtype, device=x.device
        )
        x = x * scale
        positions = torch.arange(
            past_length, past_length + length, device=input_ids.device
        )
        local = self.rotary_local(positions, x.dtype)
        glob = self.rotary_global(positions, x.dtype)
        cache: list[LayerCache] = []
        for index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[index]
            cos, sin = local if self.config.is_sliding(index) else glob
            x, present = layer(x, cos, sin, past)
            cache.append(present)
        return self.norm(x), cache


class Gemma3ForCausalLM(nn.Module):
    """Gemma 3 decoder with a language-modeling head."""

    def __init__(self, config: Gemma3Config) -> None:
        """Build the decoder and the output projection.

        Args:
            config: Model hyperparameters.
        """
        super().__init__()
        self.config = config
        self.model = Gemma3TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: list[LayerCache] | None = None,
    ) -> tuple[torch.Tensor, list[LayerCache]]:
        """Compute next-token logits for ``input_ids``.

        Args:
            input_ids: Integer tensor shaped ``(batch, length)``.
            past_key_values: Per-layer caches from earlier steps, or ``None``.

        Returns:
            Logits shaped ``(batch, length, vocab_size)`` and the caches.
        """
        hidden, cache = self.model(input_ids, past_key_values)
        return self.lm_head(hidden), cache

    @torch.no_grad()
    def generate(
        self, input_ids: torch.Tensor, max_new_tokens: int
    ) -> torch.Tensor:
        """Greedily extend ``input_ids`` using the key/value cache.

        Args:
            input_ids: Prompt token ids shaped ``(batch, length)``.
            max_new_tokens: Number of tokens to append.

        Returns:
            The prompt with the generated tokens appended.
        """
        logits, cache = self(input_ids)
        generated = input_ids
        for step in range(max_new_tokens):
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=-1)
            if step + 1 < max_new_tokens:
                logits, cache = self(next_token, cache)
        return generated


def gemma3(name: str) -> Gemma3ForCausalLM:
    """Build one of the published Gemma 3 sizes by name.

    Args:
        name: A key of :data:`~configs.gemma3.GEMMA3_CONFIGS`, such as
            ``"Gemma3-1B"``.

    Returns:
        The corresponding model.

    Raises:
        KeyError: If ``name`` is not a known size.
    """
    if name not in GEMMA3_CONFIGS:
        known = ", ".join(GEMMA3_CONFIGS)
        raise KeyError(f"unknown Gemma 3 size {name!r}; known sizes: {known}")
    return Gemma3ForCausalLM(GEMMA3_CONFIGS[name])
