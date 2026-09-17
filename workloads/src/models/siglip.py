"""SigLIP two-tower image-text model built from the operator library."""

import math
from dataclasses import dataclass

import torch
from torch import nn

from configs.siglip import (
    SIGLIP_CONFIGS,
    SiglipConfig,
    SiglipTextConfig,
    SiglipVisionConfig,
)
from operators.activation import GELU
from operators.attention import GroupedQueryAttention
from operators.convolution import Conv2d
from operators.embedding import Embedding
from operators.linear import Linear
from operators.normalization import L2Norm, LayerNorm

TowerConfig = SiglipVisionConfig | SiglipTextConfig

ACTIVATIONS = {"gelu_pytorch_tanh": GELU}


class SiglipVisionEmbeddings(nn.Module):
    """Patch projection with learned positions and no class token."""

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        dim = config.hidden_size
        self.patch_embedding = Conv2d(
            config.num_channels,
            dim,
            config.patch_size,
            stride=config.patch_size,
        )
        num_positions = (config.image_size // config.patch_size) ** 2
        self.position_embedding = Embedding(num_positions, dim)
        self.register_buffer(
            "position_ids",
            torch.arange(num_positions)[None],
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(self.patch_embedding.weight.dtype)
        patches = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        return patches + self.position_embedding(self.position_ids)


class SiglipTextEmbeddings(nn.Module):
    """Token lookup plus learned absolute positions."""

    def __init__(self, config: SiglipTextConfig) -> None:
        super().__init__()
        dim = config.hidden_size
        self.token_embedding = Embedding(config.vocab_size, dim)
        self.position_embedding = Embedding(config.max_position_embeddings, dim)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings)[None],
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = self.position_ids[:, : input_ids.shape[1]]
        return self.token_embedding(input_ids) + self.position_embedding(
            positions
        )


class SiglipAttention(nn.Module):
    """Multi-head self-attention with separate Q/K/V/output projections."""

    def __init__(self, config: TowerConfig) -> None:
        super().__init__()
        dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = dim // self.num_heads
        self.q_proj = Linear(dim, dim)
        self.k_proj = Linear(dim, dim)
        self.v_proj = Linear(dim, dim)
        self.out_proj = Linear(dim, dim)
        self.attention = GroupedQueryAttention(
            self.num_heads, self.num_heads, self.head_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        heads = (batch, length, self.num_heads, self.head_dim)
        query = self.q_proj(x).reshape(heads).transpose(1, 2)
        key = self.k_proj(x).reshape(heads).transpose(1, 2)
        value = self.v_proj(x).reshape(heads).transpose(1, 2)
        out = self.attention(query, key, value, causal=False)
        return self.out_proj(out.transpose(1, 2).reshape(batch, length, dim))


class SiglipMLP(nn.Module):
    """Two-layer feed-forward network."""

    def __init__(self, config: TowerConfig) -> None:
        super().__init__()
        self.activation_fn = ACTIVATIONS[config.hidden_act]()
        self.fc1 = Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(x)))


class SiglipEncoderLayer(nn.Module):
    """Pre-norm transformer block: attention then feed-forward."""

    def __init__(self, config: TowerConfig) -> None:
        super().__init__()
        dim, eps = config.hidden_size, config.layer_norm_eps
        self.layer_norm1 = LayerNorm(dim, eps=eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = LayerNorm(dim, eps=eps)
        self.mlp = SiglipMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class SiglipEncoder(nn.Module):
    """Stack of encoder layers."""

    def __init__(self, config: TowerConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class PackedMultiheadAttention(nn.Module):
    """Cross-attention holding ``torch.nn.MultiheadAttention``'s parameters."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        bound = 1.0 / math.sqrt(dim)
        # Q, K and V weights stacked along the output axis, as torch packs them.
        self.in_proj_weight = nn.Parameter(
            torch.empty(3 * dim, dim).uniform_(-bound, bound)
        )
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * dim))
        self.out_proj = Linear(dim, dim)
        self.attention = GroupedQueryAttention(
            num_heads, num_heads, self.head_dim
        )

    def forward(
        self, query: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        batch, query_len, dim = query.shape
        weights = self.in_proj_weight.chunk(3)
        biases = self.in_proj_bias.chunk(3)
        inputs = (query, memory, memory)
        query, key, value = (
            (torch.matmul(x, w.t()) + b)
            .reshape(batch, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
            for x, w, b in zip(inputs, weights, biases, strict=True)
        )
        out = self.attention(query, key, value, causal=False)
        return self.out_proj(out.transpose(1, 2).reshape(batch, query_len, dim))


class SiglipMultiheadAttentionPoolingHead(nn.Module):
    """Attention pooling with a learned probe query and a feed-forward tail."""

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        dim = config.hidden_size
        self.probe = nn.Parameter(torch.randn(1, 1, dim))
        self.attention = PackedMultiheadAttention(
            dim, config.num_attention_heads
        )
        self.layernorm = LayerNorm(dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probe = self.probe.expand(x.shape[0], -1, -1)
        x = self.attention(probe, x)
        x = x + self.mlp(self.layernorm(x))
        return x[:, 0]


class SiglipTextModel(nn.Module):
    """Bidirectional text tower pooled at the last token."""

    def __init__(self, config: SiglipTextConfig) -> None:
        super().__init__()
        self.embeddings = SiglipTextEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.final_layer_norm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.head = Linear(config.hidden_size, config.projection_size)

    def forward(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.final_layer_norm(self.encoder(self.embeddings(input_ids)))
        return x, self.head(x[:, -1])


class SiglipVisionModel(nn.Module):
    """Vision tower pooled by attention over all patches."""

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.head = SiglipMultiheadAttentionPoolingHead(config)

    def forward(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.post_layernorm(self.encoder(self.embeddings(pixel_values)))
        return x, self.head(x)


@dataclass(frozen=True)
class SiglipOutput:
    """Unit-normalized embeddings and the scaled, shifted similarities."""

    image_embeds: torch.Tensor
    text_embeds: torch.Tensor
    logits_per_image: torch.Tensor


class SiglipModel(nn.Module):
    """Text and vision towers sharing an embedding space."""

    def __init__(self, config: SiglipConfig) -> None:
        super().__init__()
        self.config = config
        self.text_model = SiglipTextModel(config.text)
        self.vision_model = SiglipVisionModel(config.vision)
        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))
        self.normalize = L2Norm()

    def forward(
        self, input_ids: torch.Tensor, pixel_values: torch.Tensor
    ) -> SiglipOutput:
        _, image_pooled = self.vision_model(pixel_values)
        _, text_pooled = self.text_model(input_ids)
        image_embeds = self.normalize(image_pooled)
        text_embeds = self.normalize(text_pooled)
        logits_per_text = (
            torch.matmul(text_embeds, image_embeds.t()) * self.logit_scale.exp()
            + self.logit_bias
        )
        return SiglipOutput(image_embeds, text_embeds, logits_per_text.t())


def siglip(key: str) -> SiglipModel:
    if key not in SIGLIP_CONFIGS:
        known = ", ".join(SIGLIP_CONFIGS)
        raise KeyError(f"unknown SigLIP size {key!r}; known sizes: {known}")
    return SiglipModel(SIGLIP_CONFIGS[key])
