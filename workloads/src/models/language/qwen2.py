"""Qwen2.5 causal language model built from the operator library."""

import torch
from torch import nn

from configs.language.qwen2 import QWEN2_CONFIGS, Qwen2Config
from operators.activation import SiLU
from operators.attention import GroupedQueryAttention
from operators.embedding import Embedding
from operators.linear import Linear
from operators.normalization import RMSNorm
from operators.positional import RotaryEmbedding

LayerCache = tuple[torch.Tensor, torch.Tensor]


class Qwen2Attention(nn.Module):
    """Grouped-query self-attention with RoPE and biased Q/K/V projections."""

    def __init__(self, config: Qwen2Config, rotary: RotaryEmbedding) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        kv_dim = self.num_kv_heads * self.head_dim
        bias = config.qkv_bias
        self.q_proj = Linear(config.hidden_size, config.hidden_size, bias=bias)
        self.k_proj = Linear(config.hidden_size, kv_dim, bias=bias)
        self.v_proj = Linear(config.hidden_size, kv_dim, bias=bias)
        self.o_proj = Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary = rotary
        self.attention = GroupedQueryAttention(
            self.num_heads, self.num_kv_heads, self.head_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past: LayerCache | None,
    ) -> tuple[torch.Tensor, LayerCache]:
        batch, length, hidden = x.shape
        query = self.q_proj(x).reshape(
            batch, length, self.num_heads, self.head_dim
        )
        key = self.k_proj(x).reshape(
            batch, length, self.num_kv_heads, self.head_dim
        )
        value = self.v_proj(x).reshape(
            batch, length, self.num_kv_heads, self.head_dim
        )
        query = self.rotary.apply_rotary(query.transpose(1, 2), cos, sin)
        key = self.rotary.apply_rotary(key.transpose(1, 2), cos, sin)
        value = value.transpose(1, 2)
        if past is not None:
            past_key, past_value = past
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
        out = self.attention(query, key, value)
        out = out.transpose(1, 2).reshape(batch, length, hidden)
        return self.o_proj(out), (key, value)


class Qwen2MLP(nn.Module):
    """Gated feed-forward network with a SiLU gate."""

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        hidden, inner = config.hidden_size, config.intermediate_size
        self.gate_proj = Linear(hidden, inner, bias=False)
        self.up_proj = Linear(hidden, inner, bias=False)
        self.down_proj = Linear(inner, hidden, bias=False)
        self.act_fn = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen2DecoderLayer(nn.Module):
    """Pre-norm transformer block: attention then feed-forward."""

    def __init__(self, config: Qwen2Config, rotary: RotaryEmbedding) -> None:
        super().__init__()
        self.self_attn = Qwen2Attention(config, rotary)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

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
        x = x + attended
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, present


class Qwen2Model(nn.Module):
    """Embedding table, decoder stack and final norm."""

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.rotary = RotaryEmbedding(
            config.hidden_size // config.num_attention_heads, config.rope_theta
        )
        self.layers = nn.ModuleList(
            Qwen2DecoderLayer(config, self.rotary)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        positions = torch.arange(
            past_length, past_length + length, device=input_ids.device
        )
        cos, sin = self.rotary(positions, x.dtype)
        cache: list[LayerCache] = []
        for index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[index]
            x, present = layer(x, cos, sin, past)
            cache.append(present)
        return self.norm(x), cache


class Qwen2ForCausalLM(nn.Module):
    """Qwen2.5 decoder with a language-modeling head."""

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
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
        logits, cache = self(input_ids)
        generated = input_ids
        for step in range(max_new_tokens):
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=-1)
            if step + 1 < max_new_tokens:
                logits, cache = self(next_token, cache)
        return generated


def qwen2(key: str) -> Qwen2ForCausalLM:
    if key not in QWEN2_CONFIGS:
        known = ", ".join(QWEN2_CONFIGS)
        raise KeyError(f"unknown Qwen2.5 size {key!r}; known sizes: {known}")
    return Qwen2ForCausalLM(QWEN2_CONFIGS[key])
