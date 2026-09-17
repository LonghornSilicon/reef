"""ViT image classifier built from the operator library."""

import torch
from torch import nn

from configs.vit import VIT_CONFIGS, ViTConfig
from operators.activation import ErfGELU
from operators.attention import GroupedQueryAttention
from operators.convolution import Conv2d
from operators.dropout import Dropout
from operators.interpolation import Interpolate
from operators.linear import Linear
from operators.normalization import LayerNorm


class ViTPatchEmbeddings(nn.Module):
    """Strided convolution that turns an image into patch tokens."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.projection = Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x).flatten(2).transpose(1, 2)


class ViTEmbeddings(nn.Module):
    """Class token, patch tokens and learned position embeddings."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.patch_embeddings = ViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1, config.hidden_size)
        )
        self.dropout = Dropout(0.0)

    def interpolate_pos_encoding(self, height: int, width: int) -> torch.Tensor:
        num_positions = self.position_embeddings.shape[1] - 1
        new_height = height // self.patch_size
        new_width = width // self.patch_size
        if new_height * new_width == num_positions and height == width:
            return self.position_embeddings
        cls_pos = self.position_embeddings[:, :1]
        patch_pos = self.position_embeddings[:, 1:]
        dim = patch_pos.shape[-1]
        side = int(num_positions**0.5)
        grid = patch_pos.reshape(1, side, side, dim).permute(0, 3, 1, 2)
        resize = Interpolate(size=(new_height, new_width), mode="bicubic")
        grid = resize(grid).permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((cls_pos, grid), dim=1)

    def forward(
        self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False
    ) -> torch.Tensor:
        batch, _, height, width = pixel_values.shape
        tokens = self.patch_embeddings(pixel_values)
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        if interpolate_pos_encoding:
            positions = self.interpolate_pos_encoding(height, width)
        else:
            positions = self.position_embeddings
        return self.dropout(tokens + positions)


class ViTAttention(nn.Module):
    """Multi-head self-attention with separate Q/K/V projections."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = hidden // self.num_heads
        self.q_proj = Linear(hidden, hidden, bias=config.qkv_bias)
        self.k_proj = Linear(hidden, hidden, bias=config.qkv_bias)
        self.v_proj = Linear(hidden, hidden, bias=config.qkv_bias)
        self.o_proj = Linear(hidden, hidden)
        self.attention = GroupedQueryAttention(
            self.num_heads, self.num_heads, self.head_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, hidden = x.shape
        shape = (batch, length, self.num_heads, self.head_dim)
        query = self.q_proj(x).reshape(shape).transpose(1, 2)
        key = self.k_proj(x).reshape(shape).transpose(1, 2)
        value = self.v_proj(x).reshape(shape).transpose(1, 2)
        out = self.attention(query, key, value, causal=False)
        return self.o_proj(out.transpose(1, 2).reshape(batch, length, hidden))


class ViTMLP(nn.Module):
    """Two-layer feed-forward network with an exact GELU."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.fc1 = Linear(config.hidden_size, config.intermediate_size)
        self.activation_fn = ErfGELU()
        self.fc2 = Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(x)))


class ViTLayer(nn.Module):
    """Pre-norm transformer block: attention then feed-forward."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        eps = config.layer_norm_eps
        self.attention = ViTAttention(config)
        self.layernorm_before = LayerNorm(config.hidden_size, eps=eps)
        self.layernorm_after = LayerNorm(config.hidden_size, eps=eps)
        self.mlp = ViTMLP(config)
        self.dropout = Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.layernorm_before(x)))
        return x + self.dropout(self.mlp(self.layernorm_after(x)))


class ViTModel(nn.Module):
    """Embeddings, encoder stack and final layer norm."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = ViTEmbeddings(config)
        self.layers = nn.ModuleList(
            ViTLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.layernorm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False
    ) -> torch.Tensor:
        x = self.embeddings(pixel_values, interpolate_pos_encoding)
        for layer in self.layers:
            x = layer(x)
        return self.layernorm(x)


class ViTForImageClassification(nn.Module):
    """ViT encoder with a linear head on the class token."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.vit = ViTModel(config)
        self.classifier = Linear(config.hidden_size, config.num_labels)

    def forward(
        self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False
    ) -> torch.Tensor:
        hidden = self.vit(pixel_values, interpolate_pos_encoding)
        return self.classifier(hidden[:, 0])


def vit(key: str) -> ViTForImageClassification:
    if key not in VIT_CONFIGS:
        known = ", ".join(VIT_CONFIGS)
        raise KeyError(f"unknown ViT size {key!r}; known sizes: {known}")
    return ViTForImageClassification(VIT_CONFIGS[key])
