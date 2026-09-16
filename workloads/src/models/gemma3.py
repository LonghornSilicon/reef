"""Gemma 3 causal language model built from the operator library."""

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
    """Self-attention with per-head Q/K normalization and RoPE."""

    def __init__(
        self,
        config: Gemma3Config,
        rotary: RotaryEmbedding,
        sliding: bool,
    ) -> None:
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
        super().__init__()
        hidden, inner = config.hidden_size, config.intermediate_size
        self.gate_proj = Linear(hidden, inner, bias=False)
        self.up_proj = Linear(hidden, inner, bias=False)
        self.down_proj = Linear(inner, hidden, bias=False)
        self.act_fn = GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Gemma3DecoderLayer(nn.Module):
    """Transformer block with both blocks normalized on entry and exit."""

    def __init__(
        self,
        config: Gemma3Config,
        rotary: RotaryEmbedding,
        sliding: bool,
    ) -> None:
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
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
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
        length = input_ids.shape[1]
        past_length = (
            0 if past_key_values is None else past_key_values[0][0].shape[2]
        )
        x = self.embed_tokens(input_ids)
        # The multiplier is cast to the hidden dtype before multiplying,
        # matching the reference.
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
    """Gemma 3 decoder with a language-modeling head.

    Names follow ``transformers``' ``Gemma3ForCausalLM``, so a published
    text-backbone ``state_dict`` loads with ``strict=True``.
    """

    def __init__(self, config: Gemma3Config) -> None:
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
        hidden, cache = self.model(input_ids, past_key_values)
        return self.lm_head(hidden), cache

    @torch.no_grad()
    def generate(
        self, input_ids: torch.Tensor, max_new_tokens: int
    ) -> torch.Tensor:
        """Greedy decoding with the key/value cache."""
        logits, cache = self(input_ids)
        generated = input_ids
        for step in range(max_new_tokens):
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=-1)
            if step + 1 < max_new_tokens:
                logits, cache = self(next_token, cache)
        return generated


def gemma3(name: str) -> Gemma3ForCausalLM:
    if name not in GEMMA3_CONFIGS:
        known = ", ".join(GEMMA3_CONFIGS)
        raise KeyError(f"unknown Gemma 3 size {name!r}; known sizes: {known}")
    return Gemma3ForCausalLM(GEMMA3_CONFIGS[name])
