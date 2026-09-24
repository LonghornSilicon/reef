"""GPT-Neo causal language model built from the operator library."""

import torch
from torch import nn

from workloads.configs.gpt_neo import GPT_NEO_CONFIGS, GPTNeoConfig
from workloads.operators.activation import GELU
from workloads.operators.attention import GroupedQueryAttention
from workloads.operators.embedding import Embedding
from workloads.operators.linear import Linear
from workloads.operators.normalization import LayerNorm

LayerCache = tuple[torch.Tensor, torch.Tensor]


class GPTNeoSelfAttention(nn.Module):
    """Self-attention with separate, bias-free query/key/value projections."""

    def __init__(self, config: GPTNeoConfig, attention_type: str) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.k_proj = Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = Linear(self.embed_dim, self.embed_dim)
        local = attention_type == "local"
        self.attention = GroupedQueryAttention(
            self.num_heads,
            self.num_heads,
            self.head_dim,
            # GPT-Neo scores are raw q @ k.T: it drops the 1/sqrt(head_dim)
            # factor every other decoder here applies.
            scale=1.0,
            sliding_window=config.window_size if local else None,
        )

    def forward(
        self, x: torch.Tensor, past: LayerCache | None
    ) -> tuple[torch.Tensor, LayerCache]:
        batch, length, _ = x.shape
        heads = (batch, length, self.num_heads, self.head_dim)
        query = self.q_proj(x).reshape(heads).transpose(1, 2)
        key = self.k_proj(x).reshape(heads).transpose(1, 2)
        value = self.v_proj(x).reshape(heads).transpose(1, 2)
        if past is not None:
            past_key, past_value = past
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
        out = self.attention(query, key, value)
        out = out.transpose(1, 2).reshape(batch, length, self.embed_dim)
        return self.out_proj(out), (key, value)


class GPTNeoAttention(nn.Module):
    """Holder giving the reference's ``attn.attention`` parameter names."""

    def __init__(self, config: GPTNeoConfig, layer: int) -> None:
        super().__init__()
        self.attention = GPTNeoSelfAttention(
            config, config.attention_type(layer)
        )

    def forward(
        self, x: torch.Tensor, past: LayerCache | None
    ) -> tuple[torch.Tensor, LayerCache]:
        return self.attention(x, past)


class GPTNeoMLP(nn.Module):
    """Two-layer feed-forward network with a tanh-approximated GELU."""

    def __init__(self, config: GPTNeoConfig) -> None:
        super().__init__()
        inner = (
            4 * config.hidden_size
            if config.intermediate_size is None
            else config.intermediate_size
        )
        self.c_fc = Linear(config.hidden_size, inner)
        self.c_proj = Linear(inner, config.hidden_size)
        self.act = GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))


class GPTNeoBlock(nn.Module):
    """Pre-norm transformer block: attention then feed-forward."""

    def __init__(self, config: GPTNeoConfig, layer: int) -> None:
        super().__init__()
        eps = config.layer_norm_epsilon
        self.ln_1 = LayerNorm(config.hidden_size, eps=eps)
        self.attn = GPTNeoAttention(config, layer)
        self.ln_2 = LayerNorm(config.hidden_size, eps=eps)
        self.mlp = GPTNeoMLP(config)

    def forward(
        self, x: torch.Tensor, past: LayerCache | None
    ) -> tuple[torch.Tensor, LayerCache]:
        attended, present = self.attn(self.ln_1(x), past)
        x = x + attended
        x = x + self.mlp(self.ln_2(x))
        return x, present


class GPTNeoModel(nn.Module):
    """Token and learned position embeddings, block stack and final norm."""

    def __init__(self, config: GPTNeoConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = Embedding(config.vocab_size, config.hidden_size)
        self.wpe = Embedding(config.max_position_embeddings, config.hidden_size)
        self.h = nn.ModuleList(
            GPTNeoBlock(config, layer) for layer in range(config.num_layers)
        )
        self.ln_f = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

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


class GPTNeoForCausalLM(nn.Module):
    """GPT-Neo decoder with a language-modeling head."""

    def __init__(self, config: GPTNeoConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = GPTNeoModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
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


def gpt_neo(key: str) -> GPTNeoForCausalLM:
    if key not in GPT_NEO_CONFIGS:
        known = ", ".join(GPT_NEO_CONFIGS)
        raise KeyError(f"unknown GPT-Neo size {key!r}; known sizes: {known}")
    return GPTNeoForCausalLM(GPT_NEO_CONFIGS[key])
