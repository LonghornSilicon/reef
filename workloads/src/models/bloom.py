"""BLOOM causal language model built from the operator library."""

import math

import torch
from torch import nn

from configs.bloom import BLOOM_CONFIGS, BloomConfig
from operators.activation import GELU
from operators.attention import GroupedQueryAttention
from operators.embedding import Embedding
from operators.linear import Linear
from operators.normalization import LayerNorm

LayerCache = tuple[torch.Tensor, torch.Tensor]


def alibi_slopes(num_heads: int) -> torch.Tensor:
    # 2^(-8i/n) for a power-of-two head count; otherwise the first 2^k heads
    # use that geometric series and the rest interleave the next one, as in
    # transformers' build_alibi_tensor.
    closest = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(2 ** (-(2 ** -(math.log2(closest) - 3))))
    slopes = base ** torch.arange(1, 1 + closest, dtype=torch.int32)
    if closest != num_heads:
        extra_base = torch.tensor(2 ** (-(2 ** -(math.log2(2 * closest) - 3))))
        remaining = min(closest, num_heads - closest)
        extra = torch.arange(1, 1 + 2 * remaining, 2, dtype=torch.int32)
        slopes = torch.cat((slopes, extra_base**extra))
    return slopes


def build_alibi(
    num_heads: int, key_len: int, device: torch.device
) -> torch.Tensor:
    slopes = alibi_slopes(num_heads).to(device)
    positions = torch.arange(key_len, device=device, dtype=torch.float32)
    # (1, heads, 1, key_len), fp32: broadcasts over batch and query position.
    return (slopes[:, None] * positions[None, :])[None, :, None, :]


class BloomAttention(nn.Module):
    """Multi-head self-attention with a fused, head-interleaved QKV."""

    def __init__(self, config: BloomConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.query_key_value = Linear(self.hidden_size, 3 * self.hidden_size)
        self.dense = Linear(self.hidden_size, self.hidden_size)
        self.attention = GroupedQueryAttention(
            self.num_heads, self.num_heads, self.head_dim
        )

    def forward(
        self, x: torch.Tensor, alibi: torch.Tensor, past: LayerCache | None
    ) -> tuple[torch.Tensor, LayerCache]:
        batch, length, _ = x.shape
        fused = self.query_key_value(x).reshape(
            batch, length, self.num_heads, 3, self.head_dim
        )
        query = fused[..., 0, :].transpose(1, 2)
        key = fused[..., 1, :].transpose(1, 2)
        value = fused[..., 2, :].transpose(1, 2)
        if past is not None:
            past_key, past_value = past
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
        out = self.attention(query, key, value, bias=alibi)
        out = out.transpose(1, 2).reshape(batch, length, self.hidden_size)
        return self.dense(out), (key, value)


class BloomMLP(nn.Module):
    """Two-layer feed-forward network with a tanh-approximated GELU."""

    def __init__(self, config: BloomConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.dense_h_to_4h = Linear(hidden, 4 * hidden)
        self.gelu_impl = GELU()
        self.dense_4h_to_h = Linear(4 * hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense_4h_to_h(self.gelu_impl(self.dense_h_to_4h(x)))


class BloomBlock(nn.Module):
    """Pre-norm transformer block: attention then feed-forward."""

    def __init__(self, config: BloomConfig) -> None:
        super().__init__()
        eps = config.layer_norm_epsilon
        self.input_layernorm = LayerNorm(config.hidden_size, eps=eps)
        self.self_attention = BloomAttention(config)
        self.post_attention_layernorm = LayerNorm(config.hidden_size, eps=eps)
        self.mlp = BloomMLP(config)
        self.post_layernorm_residual = (
            config.apply_residual_connection_post_layernorm
        )

    def forward(
        self, x: torch.Tensor, alibi: torch.Tensor, past: LayerCache | None
    ) -> tuple[torch.Tensor, LayerCache]:
        normed = self.input_layernorm(x)
        residual = normed if self.post_layernorm_residual else x
        attended, present = self.self_attention(normed, alibi, past)
        x = residual + attended
        normed = self.post_attention_layernorm(x)
        residual = normed if self.post_layernorm_residual else x
        return residual + self.mlp(normed), present


class BloomModel(nn.Module):
    """Normalized embedding table, block stack and final norm."""

    def __init__(self, config: BloomConfig) -> None:
        super().__init__()
        self.config = config
        eps = config.layer_norm_epsilon
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size)
        self.word_embeddings_layernorm = LayerNorm(config.hidden_size, eps=eps)
        self.h = nn.ModuleList(
            BloomBlock(config) for _ in range(config.n_layer)
        )
        self.ln_f = LayerNorm(config.hidden_size, eps=eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: list[LayerCache] | None = None,
    ) -> tuple[torch.Tensor, list[LayerCache]]:
        length = input_ids.shape[1]
        past_length = (
            0 if past_key_values is None else past_key_values[0][0].shape[2]
        )
        x = self.word_embeddings_layernorm(self.word_embeddings(input_ids))
        alibi = build_alibi(
            self.config.n_head, past_length + length, input_ids.device
        ).to(x.dtype)
        cache: list[LayerCache] = []
        for index, block in enumerate(self.h):
            past = None if past_key_values is None else past_key_values[index]
            x, present = block(x, alibi, past)
            cache.append(present)
        return self.ln_f(x), cache


class BloomForCausalLM(nn.Module):
    """BLOOM decoder with a language-modeling head."""

    def __init__(self, config: BloomConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = BloomModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.word_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: list[LayerCache] | None = None,
    ) -> tuple[torch.Tensor, list[LayerCache]]:
        hidden, cache = self.transformer(input_ids, past_key_values)
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


def bloom(key: str) -> BloomForCausalLM:
    if key not in BLOOM_CONFIGS:
        known = ", ".join(BLOOM_CONFIGS)
        raise KeyError(f"unknown BLOOM size {key!r}; known sizes: {known}")
    return BloomForCausalLM(BLOOM_CONFIGS[key])
