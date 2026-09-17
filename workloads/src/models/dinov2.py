"""DINOv2 image classifier built from the operator library."""

import torch
from torch import nn

from configs.dinov2 import DINOV2_CONFIGS, Dinov2Config
from operators.activation import ErfGELU, SiLU
from operators.attention import GroupedQueryAttention
from operators.convolution import Conv2d
from operators.dropout import Dropout
from operators.interpolation import Interpolate
from operators.linear import Linear
from operators.normalization import LayerNorm


class Dinov2PatchEmbeddings(nn.Module):
    """Strided convolution that turns an image into patch tokens."""

    def __init__(self, config: Dinov2Config) -> None:
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


class Dinov2Embeddings(nn.Module):
    """Class token, patch tokens and always-interpolated position embeddings."""

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        # Only used for masked pretraining, but the checkpoint carries it.
        self.mask_token = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.patch_embeddings = Dinov2PatchEmbeddings(config)
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
        # transformers resamples in float32 whatever the parameter dtype.
        grid = resize(grid.to(torch.float32)).to(patch_pos.dtype)
        grid = grid.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((cls_pos, grid), dim=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = pixel_values.shape
        tokens = self.patch_embeddings(pixel_values)
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        return self.dropout(
            tokens + self.interpolate_pos_encoding(height, width)
        )


class Dinov2SelfAttention(nn.Module):
    """Multi-head self-attention with separate Q/K/V projections."""

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = hidden // self.num_heads
        self.query = Linear(hidden, hidden, bias=config.qkv_bias)
        self.key = Linear(hidden, hidden, bias=config.qkv_bias)
        self.value = Linear(hidden, hidden, bias=config.qkv_bias)
        self.attention = GroupedQueryAttention(
            self.num_heads, self.num_heads, self.head_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, hidden = x.shape
        shape = (batch, length, self.num_heads, self.head_dim)
        query = self.query(x).reshape(shape).transpose(1, 2)
        key = self.key(x).reshape(shape).transpose(1, 2)
        value = self.value(x).reshape(shape).transpose(1, 2)
        out = self.attention(query, key, value, causal=False)
        return out.transpose(1, 2).reshape(batch, length, hidden)


class Dinov2SelfOutput(nn.Module):
    """Output projection of the attention block."""

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.dense = Linear(config.hidden_size, config.hidden_size)
        self.dropout = Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.dense(x))


class Dinov2Attention(nn.Module):
    """Self-attention followed by its output projection."""

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.attention = Dinov2SelfAttention(config)
        self.output = Dinov2SelfOutput(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.attention(x))


class Dinov2LayerScale(nn.Module):
    """Per-channel learned gain on a residual branch."""

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(
            config.layerscale_value * torch.ones(config.hidden_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.lambda1


class Dinov2MLP(nn.Module):
    """Two-layer feed-forward network with an exact GELU."""

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        inner = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = Linear(config.hidden_size, inner)
        self.activation = ErfGELU()
        self.fc2 = Linear(inner, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class Dinov2SwiGLUFFN(nn.Module):
    """Gated feed-forward network with one fused input projection."""

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        inner = int(config.hidden_size * config.mlp_ratio)
        inner = (int(inner * 2 / 3) + 7) // 8 * 8
        self.weights_in = Linear(config.hidden_size, 2 * inner)
        self.act = SiLU()
        self.weights_out = Linear(inner, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.weights_in(x).chunk(2, dim=-1)
        return self.weights_out(self.act(gate) * up)


class Dinov2Layer(nn.Module):
    """Pre-norm transformer block with LayerScale on both branches."""

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        eps = config.layer_norm_eps
        self.norm1 = LayerNorm(config.hidden_size, eps=eps)
        self.attention = Dinov2Attention(config)
        self.layer_scale1 = Dinov2LayerScale(config)
        self.norm2 = LayerNorm(config.hidden_size, eps=eps)
        if config.use_swiglu_ffn:
            self.mlp = Dinov2SwiGLUFFN(config)
        else:
            self.mlp = Dinov2MLP(config)
        self.layer_scale2 = Dinov2LayerScale(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.layer_scale1(self.attention(self.norm1(x)))
        return x + self.layer_scale2(self.mlp(self.norm2(x)))


class Dinov2Encoder(nn.Module):
    """Stack of transformer blocks."""

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            Dinov2Layer(config) for _ in range(config.num_hidden_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layer:
            x = layer(x)
        return x


class Dinov2Model(nn.Module):
    """Embeddings, encoder and final layer norm."""

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.config = config
        self.embeddings = Dinov2Embeddings(config)
        self.encoder = Dinov2Encoder(config)
        self.layernorm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.layernorm(self.encoder(self.embeddings(pixel_values)))


class Dinov2ForImageClassification(nn.Module):
    """DINOv2 encoder with a linear head on cat(cls, mean of patch tokens)."""

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.config = config
        self.dinov2 = Dinov2Model(config)
        self.classifier = Linear(config.hidden_size * 2, config.num_labels)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden = self.dinov2(pixel_values)
        pooled = torch.cat((hidden[:, 0], hidden[:, 1:].mean(dim=1)), dim=1)
        return self.classifier(pooled)


def dinov2(key: str) -> Dinov2ForImageClassification:
    if key not in DINOV2_CONFIGS:
        known = ", ".join(DINOV2_CONFIGS)
        raise KeyError(f"unknown DINOv2 size {key!r}; known sizes: {known}")
    return Dinov2ForImageClassification(DINOV2_CONFIGS[key])
