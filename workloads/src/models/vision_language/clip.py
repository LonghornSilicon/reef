"""CLIP two-tower image-text model built from the operator library."""

from dataclasses import dataclass

import torch
from torch import nn

from configs.vision_language.clip import (
    CLIP_CONFIGS,
    CLIPConfig,
    CLIPTextConfig,
    CLIPVisionConfig,
)
from operators.activation import Sigmoid
from operators.attention import GroupedQueryAttention
from operators.convolution import Conv2d
from operators.embedding import Embedding
from operators.linear import Linear
from operators.normalization import L2Norm, LayerNorm

TowerConfig = CLIPVisionConfig | CLIPTextConfig


class QuickGELU(nn.Module):
    """OpenAI CLIP's ``x * sigmoid(1.702 x)`` stand-in for GELU."""

    def __init__(self) -> None:
        super().__init__()
        self.sigmoid = Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(1.702 * x)


ACTIVATIONS = {"quick_gelu": QuickGELU}


class CLIPVisionEmbeddings(nn.Module):
    """Patch projection with a class token and learned positions."""

    def __init__(self, config: CLIPVisionConfig) -> None:
        super().__init__()
        dim = config.hidden_size
        self.class_embedding = nn.Parameter(torch.randn(dim))
        self.patch_embedding = Conv2d(
            config.num_channels,
            dim,
            config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        num_positions = (config.image_size // config.patch_size) ** 2 + 1
        self.position_embedding = Embedding(num_positions, dim)
        self.register_buffer(
            "position_ids",
            torch.arange(num_positions)[None],
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(self.patch_embedding.weight.dtype)
        patches = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        batch = patches.shape[0]
        class_token = self.class_embedding.expand(batch, 1, -1)
        x = torch.cat((class_token, patches), dim=1)
        return x + self.position_embedding(self.position_ids)


class CLIPTextEmbeddings(nn.Module):
    """Token lookup plus learned absolute positions."""

    def __init__(self, config: CLIPTextConfig) -> None:
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


class CLIPAttention(nn.Module):
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

    def forward(self, x: torch.Tensor, causal: bool) -> torch.Tensor:
        batch, length, dim = x.shape
        heads = (batch, length, self.num_heads, self.head_dim)
        query = self.q_proj(x).reshape(heads).transpose(1, 2)
        key = self.k_proj(x).reshape(heads).transpose(1, 2)
        value = self.v_proj(x).reshape(heads).transpose(1, 2)
        out = self.attention(query, key, value, causal=causal)
        return self.out_proj(out.transpose(1, 2).reshape(batch, length, dim))


class CLIPMLP(nn.Module):
    """Two-layer feed-forward network."""

    def __init__(self, config: TowerConfig) -> None:
        super().__init__()
        self.activation_fn = ACTIVATIONS[config.hidden_act]()
        self.fc1 = Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(x)))


class CLIPEncoderLayer(nn.Module):
    """Pre-norm transformer block: attention then feed-forward."""

    def __init__(self, config: TowerConfig) -> None:
        super().__init__()
        dim, eps = config.hidden_size, config.layer_norm_eps
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = LayerNorm(dim, eps=eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, causal: bool) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x), causal)
        return x + self.mlp(self.layer_norm2(x))


class CLIPEncoder(nn.Module):
    """Stack of encoder layers."""

    def __init__(self, config: TowerConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)
        )

    def forward(self, x: torch.Tensor, causal: bool) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, causal)
        return x


class CLIPTextModel(nn.Module):
    """Causal text tower pooled at the first EOS token."""

    def __init__(self, config: CLIPTextConfig) -> None:
        super().__init__()
        self.eos_token_id = config.eos_token_id
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(self.embeddings(input_ids), causal=True)
        x = self.final_layer_norm(x)
        rows = torch.arange(x.shape[0], device=x.device)
        eos = (input_ids == self.eos_token_id).int().argmax(dim=-1)
        return x, x[rows, eos]


class CLIPVisionModel(nn.Module):
    """Bidirectional vision tower pooled at the class token."""

    def __init__(self, config: CLIPVisionConfig) -> None:
        super().__init__()
        dim, eps = config.hidden_size, config.layer_norm_eps
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = LayerNorm(dim, eps=eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = LayerNorm(dim, eps=eps)

    def forward(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.pre_layrnorm(self.embeddings(pixel_values))
        x = self.encoder(x, causal=False)
        return x, self.post_layernorm(x[:, 0])


@dataclass(frozen=True)
class CLIPOutput:
    """Unit-normalized embeddings and the scaled cosine similarities."""

    image_embeds: torch.Tensor
    text_embeds: torch.Tensor
    logits_per_image: torch.Tensor


class CLIPModel(nn.Module):
    """Text and vision towers projected into a shared embedding space."""

    def __init__(self, config: CLIPConfig) -> None:
        super().__init__()
        self.config = config
        self.text_model = CLIPTextModel(config.text)
        self.vision_model = CLIPVisionModel(config.vision)
        self.visual_projection = Linear(
            config.vision.hidden_size, config.projection_dim, bias=False
        )
        self.text_projection = Linear(
            config.text.hidden_size, config.projection_dim, bias=False
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(config.logit_scale_init_value)
        )
        self.normalize = L2Norm()

    def forward(
        self, input_ids: torch.Tensor, pixel_values: torch.Tensor
    ) -> CLIPOutput:
        _, image_pooled = self.vision_model(pixel_values)
        _, text_pooled = self.text_model(input_ids)
        image_embeds = self.normalize(self.visual_projection(image_pooled))
        text_embeds = self.normalize(self.text_projection(text_pooled))
        logits_per_text = (
            torch.matmul(text_embeds, image_embeds.t()) * self.logit_scale.exp()
        )
        return CLIPOutput(image_embeds, text_embeds, logits_per_text.t())


def clip(key: str) -> CLIPModel:
    if key not in CLIP_CONFIGS:
        known = ", ".join(CLIP_CONFIGS)
        raise KeyError(f"unknown CLIP size {key!r}; known sizes: {known}")
    return CLIPModel(CLIP_CONFIGS[key])
