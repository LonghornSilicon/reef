"""MobileViT image classifier built from the operator library."""

import math

import torch
from torch import nn

from configs.mobilevit import MOBILEVIT_CONFIGS, MobileViTConfig
from operators.activation import SiLU
from operators.attention import GroupedQueryAttention
from operators.convolution import Conv2d
from operators.dropout import Dropout
from operators.interpolation import Interpolate
from operators.linear import Linear
from operators.normalization import BatchNorm2d, LayerNorm

HIDDEN_DROPOUT = 0.1
CLASSIFIER_DROPOUT = 0.1

# Blocks per encoder layer and which of them are MobileViT (transformer)
# layers, as transformers' MobileViTEncoder hard-codes them.
MOBILENET_DEPTHS = (1, 3)
TRANSFORMER_DEPTHS = (2, 4, 3)


def make_divisible(value: int, divisor: int = 8) -> int:
    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * value:
        rounded += divisor
    return rounded


class MobileViTConvLayer(nn.Module):
    """Convolution with optional batch norm and SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        use_normalization: bool = True,
        use_activation: bool = True,
    ) -> None:
        super().__init__()
        self.convolution = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False,
        )
        self.normalization = (
            BatchNorm2d(out_channels) if use_normalization else None
        )
        self.activation = SiLU() if use_activation else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolution(x)
        if self.normalization is not None:
            x = self.normalization(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class MobileViTInvertedResidual(nn.Module):
    """MobileNetV2 block: expand 1x1, depthwise 3x3, reduce 1x1."""

    def __init__(
        self,
        config: MobileViTConfig,
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> None:
        super().__init__()
        expanded = make_divisible(round(in_channels * config.expand_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels
        self.expand_1x1 = MobileViTConvLayer(in_channels, expanded, 1)
        self.conv_3x3 = MobileViTConvLayer(
            expanded, expanded, 3, stride=stride, groups=expanded
        )
        self.reduce_1x1 = MobileViTConvLayer(
            expanded, out_channels, 1, use_activation=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.reduce_1x1(self.conv_3x3(self.expand_1x1(x)))
        return x + out if self.use_residual else out


class MobileViTMobileNetLayer(nn.Module):
    """Sequence of inverted residual blocks, the first carrying the stride."""

    def __init__(
        self,
        config: MobileViTConfig,
        in_channels: int,
        out_channels: int,
        stride: int,
        num_stages: int,
    ) -> None:
        super().__init__()
        self.layer = nn.ModuleList()
        for index in range(num_stages):
            self.layer.append(
                MobileViTInvertedResidual(
                    config,
                    in_channels,
                    out_channels,
                    stride if index == 0 else 1,
                )
            )
            in_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.layer:
            x = block(x)
        return x


class MobileViTSelfAttention(nn.Module):
    """Multi-head self-attention with separate Q/K/V projections."""

    def __init__(self, config: MobileViTConfig, hidden_size: int) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = hidden_size // self.num_heads
        self.query = Linear(hidden_size, hidden_size)
        self.key = Linear(hidden_size, hidden_size)
        self.value = Linear(hidden_size, hidden_size)
        self.dropout = Dropout(0.0)
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


class MobileViTSelfOutput(nn.Module):
    """Output projection of the attention block."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.dense = Linear(hidden_size, hidden_size)
        self.dropout = Dropout(HIDDEN_DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.dense(x))


class MobileViTAttention(nn.Module):
    """Self-attention followed by its output projection."""

    def __init__(self, config: MobileViTConfig, hidden_size: int) -> None:
        super().__init__()
        self.attention = MobileViTSelfAttention(config, hidden_size)
        self.output = MobileViTSelfOutput(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.attention(x))


class MobileViTIntermediate(nn.Module):
    """First feed-forward projection with SiLU."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.dense = Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.intermediate_act_fn(self.dense(x))


class MobileViTOutput(nn.Module):
    """Second feed-forward projection plus the residual."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.dense = Linear(intermediate_size, hidden_size)
        self.dropout = Dropout(HIDDEN_DROPOUT)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.dense(x)) + residual


class MobileViTTransformerLayer(nn.Module):
    """Pre-norm transformer block: attention then feed-forward."""

    def __init__(
        self, config: MobileViTConfig, hidden_size: int, intermediate_size: int
    ) -> None:
        super().__init__()
        eps = config.layer_norm_eps
        self.attention = MobileViTAttention(config, hidden_size)
        self.intermediate = MobileViTIntermediate(
            hidden_size, intermediate_size
        )
        self.output = MobileViTOutput(hidden_size, intermediate_size)
        self.layernorm_before = LayerNorm(hidden_size, eps=eps)
        self.layernorm_after = LayerNorm(hidden_size, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.layernorm_before(x))
        return self.output(self.intermediate(self.layernorm_after(x)), x)


class MobileViTTransformer(nn.Module):
    """Stack of transformer blocks."""

    def __init__(
        self, config: MobileViTConfig, hidden_size: int, num_stages: int
    ) -> None:
        super().__init__()
        intermediate_size = int(hidden_size * config.mlp_ratio)
        self.layer = nn.ModuleList(
            MobileViTTransformerLayer(config, hidden_size, intermediate_size)
            for _ in range(num_stages)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.layer:
            x = block(x)
        return x


class MobileViTLayer(nn.Module):
    """MobileViT block: local convs, patch-wise transformer, fusion conv."""

    def __init__(
        self,
        config: MobileViTConfig,
        in_channels: int,
        out_channels: int,
        hidden_size: int,
        num_stages: int,
    ) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.downsampling_layer = MobileViTInvertedResidual(
            config, in_channels, out_channels, stride=2
        )
        in_channels = out_channels
        kernel = config.conv_kernel_size
        self.conv_kxk = MobileViTConvLayer(in_channels, in_channels, kernel)
        self.conv_1x1 = MobileViTConvLayer(
            in_channels,
            hidden_size,
            1,
            use_normalization=False,
            use_activation=False,
        )
        self.transformer = MobileViTTransformer(config, hidden_size, num_stages)
        self.layernorm = LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.conv_projection = MobileViTConvLayer(hidden_size, in_channels, 1)
        self.fusion = MobileViTConvLayer(2 * in_channels, in_channels, kernel)

    def unfolding(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B * p * p, patches, C): each of the p * p pixel
        # offsets within a patch becomes its own sequence over the patches.
        batch, channels, height, width = x.shape
        p = self.patch_size
        rows, cols = math.ceil(height / p), math.ceil(width / p)
        if (rows * p, cols * p) != (height, width):
            x = Interpolate(size=(rows * p, cols * p), mode="bilinear")(x)
        x = x.reshape(batch * channels * rows, p, cols, p).transpose(1, 2)
        x = x.reshape(batch, channels, rows * cols, p * p).transpose(1, 3)
        return x.reshape(batch * p * p, rows * cols, channels)

    def folding(self, patches: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        batch, channels, height, width = shape
        p = self.patch_size
        rows, cols = math.ceil(height / p), math.ceil(width / p)
        x = patches.reshape(batch, p * p, rows * cols, channels).transpose(1, 3)
        x = x.reshape(batch * channels * rows, cols, p, p).transpose(1, 2)
        x = x.reshape(batch, channels, rows * p, cols * p)
        if (rows * p, cols * p) != (height, width):
            x = Interpolate(size=(height, width), mode="bilinear")(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsampling_layer(x)
        residual = x
        x = self.conv_1x1(self.conv_kxk(x))
        patches = self.layernorm(self.transformer(self.unfolding(x)))
        x = self.conv_projection(self.folding(patches, x.shape))
        return self.fusion(torch.cat((residual, x), dim=1))


class MobileViTEncoder(nn.Module):
    """Two MobileNet layers followed by three MobileViT layers."""

    def __init__(self, config: MobileViTConfig) -> None:
        super().__init__()
        necks = config.neck_hidden_sizes
        self.layer = nn.ModuleList()
        for index, depth in enumerate(MOBILENET_DEPTHS):
            self.layer.append(
                MobileViTMobileNetLayer(
                    config,
                    necks[index],
                    necks[index + 1],
                    stride=1 if index == 0 else 2,
                    num_stages=depth,
                )
            )
        offset = len(MOBILENET_DEPTHS)
        for index, depth in enumerate(TRANSFORMER_DEPTHS):
            self.layer.append(
                MobileViTLayer(
                    config,
                    necks[offset + index],
                    necks[offset + index + 1],
                    config.hidden_sizes[index],
                    depth,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layer:
            x = layer(x)
        return x


class MobileViTModel(nn.Module):
    """Conv stem, encoder, 1x1 expansion and global average pooling."""

    def __init__(self, config: MobileViTConfig) -> None:
        super().__init__()
        self.config = config
        necks = config.neck_hidden_sizes
        self.conv_stem = MobileViTConvLayer(3, necks[0], 3, stride=2)
        self.encoder = MobileViTEncoder(config)
        self.conv_1x1_exp = MobileViTConvLayer(necks[5], necks[6], 1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.conv_1x1_exp(self.encoder(self.conv_stem(pixel_values)))
        return x.mean(dim=(-2, -1))


class MobileViTForImageClassification(nn.Module):
    """MobileViT backbone with a dropout and linear head on pooled features."""

    def __init__(self, config: MobileViTConfig) -> None:
        super().__init__()
        self.config = config
        self.mobilevit = MobileViTModel(config)
        self.dropout = Dropout(CLASSIFIER_DROPOUT)
        self.classifier = Linear(
            config.neck_hidden_sizes[-1], config.num_labels
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.mobilevit(pixel_values)))


def mobilevit(key: str) -> MobileViTForImageClassification:
    if key not in MOBILEVIT_CONFIGS:
        known = ", ".join(MOBILEVIT_CONFIGS)
        raise KeyError(f"unknown MobileViT size {key!r}; known sizes: {known}")
    return MobileViTForImageClassification(MOBILEVIT_CONFIGS[key])
