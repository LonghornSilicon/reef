"""GPT-2 causal language model built from the operator library."""

import torch
from torch import nn

from configs.gpt2 import GPT2_CONFIGS, GPT2Config
from operators.activation import GELU
from operators.attention import GroupedQueryAttention
from operators.embedding import Embedding
from operators.linear import Linear
from operators.normalization import LayerNorm

LayerCache = tuple[torch.Tensor, torch.Tensor]


class Conv1D(nn.Module):
    """Linear map whose weight is stored transposed, ``(in, out)``, as GPT-2."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(in_features, out_features).normal_(std=0.02)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight) + self.bias


class GPT2Attention(nn.Module):
    """Multi-head self-attention with a fused query/key/value projection."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.c_attn = Conv1D(self.embed_dim, 3 * self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        self.attention = GroupedQueryAttention(
            self.num_heads,
            self.num_heads,
            self.head_dim,
            scale=self.head_dim**-0.5 if config.scale_attn_weights else 1.0,
        )

    def forward(
        self, x: torch.Tensor, past: LayerCache | None
    ) -> tuple[torch.Tensor, LayerCache]:
        batch, length, _ = x.shape
        query, key, value = self.c_attn(x).split(self.embed_dim, dim=2)
        heads = (batch, length, self.num_heads, self.head_dim)
        query = query.reshape(heads).transpose(1, 2)
        key = key.reshape(heads).transpose(1, 2)
        value = value.reshape(heads).transpose(1, 2)
        if past is not None:
            past_key, past_value = past
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
        out = self.attention(query, key, value)
        out = out.transpose(1, 2).reshape(batch, length, self.embed_dim)
        return self.c_proj(out), (key, value)


class GPT2MLP(nn.Module):
    """Two-layer feed-forward network with a tanh-approximated GELU."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        inner = 4 * config.n_embd if config.n_inner is None else config.n_inner
        self.c_fc = Conv1D(config.n_embd, inner)
        self.c_proj = Conv1D(inner, config.n_embd)
        self.act = GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))


class GPT2Block(nn.Module):
    """Pre-norm transformer block: attention then feed-forward."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(
        self, x: torch.Tensor, past: LayerCache | None
    ) -> tuple[torch.Tensor, LayerCache]:
        attended, present = self.attn(self.ln_1(x), past)
        x = x + attended
        x = x + self.mlp(self.ln_2(x))
        return x, present


class GPT2Model(nn.Module):
    """Token and learned position embeddings, block stack and final norm."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config
        self.wte = Embedding(config.vocab_size, config.n_embd)
        self.wpe = Embedding(config.n_positions, config.n_embd)
        self.h = nn.ModuleList(GPT2Block(config) for _ in range(config.n_layer))
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: list[LayerCache] | None = None,
    ) -> tuple[torch.Tensor, list[LayerCache]]:
        length = input_ids.shape[1]
        past_length = (
            0 if past_key_values is None else past_key_values[0][0].shape[2]
        )
        positions = torch.arange(
            past_length, past_length + length, device=input_ids.device
        )
        x = self.wte(input_ids) + self.wpe(positions)
        cache: list[LayerCache] = []
        for index, block in enumerate(self.h):
            past = None if past_key_values is None else past_key_values[index]
            x, present = block(x, past)
            cache.append(present)
        return self.ln_f(x), cache


class GPT2LMHeadModel(nn.Module):
    """GPT-2 decoder with a language-modeling head."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.wte.weight

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


def gpt2(key: str) -> GPT2LMHeadModel:
    if key not in GPT2_CONFIGS:
        known = ", ".join(GPT2_CONFIGS)
        raise KeyError(f"unknown GPT-2 size {key!r}; known sizes: {known}")
    return GPT2LMHeadModel(GPT2_CONFIGS[key])
