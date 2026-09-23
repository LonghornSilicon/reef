"""Llama causal language model built from the operator library."""

import torch
from torch import nn

from workloads.configs.llama import LLAMA_CONFIGS, LlamaConfig
from workloads.operators.activation import SiLU
from workloads.operators.attention import GroupedQueryAttention
from workloads.operators.embedding import Embedding
from workloads.operators.linear import Linear
from workloads.operators.normalization import RMSNorm
from workloads.operators.positional import RotaryEmbedding

LayerCache = tuple[torch.Tensor, torch.Tensor]


class LlamaAttention(nn.Module):
    """Grouped-query self-attention with RoPE."""

    def __init__(self, config: LlamaConfig, rotary: RotaryEmbedding) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        query_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        bias = config.attention_bias
        self.q_proj = Linear(config.hidden_size, query_dim, bias=bias)
        self.k_proj = Linear(config.hidden_size, kv_dim, bias=bias)
        self.v_proj = Linear(config.hidden_size, kv_dim, bias=bias)
        self.o_proj = Linear(query_dim, config.hidden_size, bias=bias)
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
        query = self.rotary.apply_rotary(query.transpose(1, 2), cos, sin)
        key = self.rotary.apply_rotary(key.transpose(1, 2), cos, sin)
        value = value.transpose(1, 2)
        if past is not None:
            past_key, past_value = past
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
        out = self.attention(query, key, value)
        out = out.transpose(1, 2).reshape(
            batch, length, self.num_heads * self.head_dim
        )
        return self.o_proj(out), (key, value)


class LlamaMLP(nn.Module):
    """Gated feed-forward network with a SiLU gate."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        hidden, inner = config.hidden_size, config.intermediate_size
        bias = config.mlp_bias
        self.gate_proj = Linear(hidden, inner, bias=bias)
        self.up_proj = Linear(hidden, inner, bias=bias)
        self.down_proj = Linear(inner, hidden, bias=bias)
        self.act_fn = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    """Pre-norm transformer block: attention then feed-forward."""

    def __init__(self, config: LlamaConfig, rotary: RotaryEmbedding) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(config, rotary)
        self.mlp = LlamaMLP(config)
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


class LlamaModel(nn.Module):
    """Embedding table, decoder stack and final norm."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.rotary = RotaryEmbedding(config.head_dim, config.rope_theta)
        self.layers = nn.ModuleList(
            LlamaDecoderLayer(config, self.rotary)
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


class LlamaForCausalLM(nn.Module):
    """Llama decoder with a language-modeling head."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
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


def llama(key: str) -> LlamaForCausalLM:
    if key not in LLAMA_CONFIGS:
        known = ", ".join(LLAMA_CONFIGS)
        raise KeyError(f"unknown Llama size {key!r}; known sizes: {known}")
    return LlamaForCausalLM(LLAMA_CONFIGS[key])
