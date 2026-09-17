"""SegFormer semantic segmenter built from the operator library."""

import torch
from torch import nn

from configs.vision_transformer.segformer import (
    SEGFORMER_CONFIGS,
    SegformerConfig,
)
from operators.activation import ErfGELU, ReLU
from operators.attention import GroupedQueryAttention
from operators.convolution import Conv2d
from operators.dropout import Dropout
from operators.interpolation import Interpolate
from operators.linear import Linear
from operators.normalization import BatchNorm2d, LayerNorm

CLASSIFIER_DROPOUT = 0.1

# transformers builds every encoder LayerNorm with nn.LayerNorm's default
# eps (1e-5) and never reads config.layer_norm_eps, so neither do we.


def tokens_to_map(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    batch, _, channels = x.shape
    return x.transpose(1, 2).reshape(batch, channels, height, width)


def map_to_tokens(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(2).transpose(1, 2)


class SegformerOverlapPatchEmbeddings(nn.Module):
    """Overlapping strided convolution followed by a layer norm on tokens."""

    def __init__(
        self, patch_size: int, stride: int, in_channels: int, hidden_size: int
    ) -> None:
        super().__init__()
        self.proj = Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.layer_norm = LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, height, width = x.shape
        return self.layer_norm(map_to_tokens(x)), height, width


class SegformerSequenceReduction(nn.Module):
    """Strided convolution that shrinks the key/value token grid."""

    def __init__(self, hidden_size: int, ratio: int) -> None:
        super().__init__()
        self.sequence_reduction = Conv2d(
            hidden_size, hidden_size, kernel_size=ratio, stride=ratio
        )
        self.layer_norm = LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        reduced = self.sequence_reduction(tokens_to_map(x, height, width))
        return self.layer_norm(map_to_tokens(reduced))


class SegformerAttention(nn.Module):
    """Self-attention whose keys and values come from a reduced grid."""

    def __init__(self, hidden_size: int, num_heads: int, ratio: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = Linear(hidden_size, hidden_size)
        self.k_proj = Linear(hidden_size, hidden_size)
        self.v_proj = Linear(hidden_size, hidden_size)
        self.o_proj = Linear(hidden_size, hidden_size)
        self.sequence_reduction = (
            SegformerSequenceReduction(hidden_size, ratio)
            if ratio > 1
            else None
        )
        self.attention = GroupedQueryAttention(
            num_heads, num_heads, self.head_dim
        )

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        shape = (batch, length, self.num_heads, self.head_dim)
        return x.reshape(shape).transpose(1, 2)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, length, hidden = x.shape
        query = self.split_heads(self.q_proj(x))
        kv = x
        if self.sequence_reduction is not None:
            kv = self.sequence_reduction(x, height, width)
        key = self.split_heads(self.k_proj(kv))
        value = self.split_heads(self.v_proj(kv))
        out = self.attention(query, key, value, causal=False)
        return self.o_proj(out.transpose(1, 2).reshape(batch, length, hidden))


class SegformerDepthWiseConv(nn.Module):
    """Depthwise 3x3 convolution applied to a token sequence."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return map_to_tokens(self.dwconv(tokens_to_map(x, height, width)))


class SegformerMixMLP(nn.Module):
    """Mix-FFN: linear, depthwise convolution, GELU, linear."""

    def __init__(self, hidden_size: int, inner: int) -> None:
        super().__init__()
        self.fc1 = Linear(hidden_size, inner)
        self.dwconv = SegformerDepthWiseConv(inner)
        self.activation_fn = ErfGELU()
        self.fc2 = Linear(inner, hidden_size)
        self.dropout = Dropout(0.0)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = self.dwconv(self.fc1(x), height, width)
        x = self.dropout(self.activation_fn(x))
        return self.dropout(self.fc2(x))


class SegformerLayer(nn.Module):
    """Pre-norm transformer block with efficient attention and Mix-FFN."""

    def __init__(
        self, hidden_size: int, num_heads: int, ratio: int, mlp_ratio: int
    ) -> None:
        super().__init__()
        self.layernorm_before = LayerNorm(hidden_size)
        self.attention = SegformerAttention(hidden_size, num_heads, ratio)
        self.layernorm_after = LayerNorm(hidden_size)
        self.mlp = SegformerMixMLP(hidden_size, int(hidden_size * mlp_ratio))
        self.hidden_dropout = Dropout(0.0)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        attended = self.attention(self.layernorm_before(x), height, width)
        x = x + self.hidden_dropout(attended)
        return x + self.mlp(self.layernorm_after(x), height, width)


class SegformerStage(nn.Module):
    """Patch embedding, transformer blocks and a layer norm for one stage."""

    def __init__(self, config: SegformerConfig, index: int) -> None:
        super().__init__()
        hidden_size = config.hidden_sizes[index]
        in_channels = 3 if index == 0 else config.hidden_sizes[index - 1]
        self.patch_embeddings = SegformerOverlapPatchEmbeddings(
            config.patch_sizes[index],
            config.strides[index],
            in_channels,
            hidden_size,
        )
        self.blocks = nn.ModuleList(
            SegformerLayer(
                hidden_size,
                config.num_attention_heads[index],
                config.sr_ratios[index],
                config.mlp_ratios[index],
            )
            for _ in range(config.depths[index])
        )
        self.layer_norm = LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, height, width = self.patch_embeddings(x)
        for block in self.blocks:
            x = block(x, height, width)
        return tokens_to_map(self.layer_norm(x), height, width)


class SegformerModel(nn.Module):
    """Hierarchical MiT encoder returning every stage's feature map."""

    def __init__(self, config: SegformerConfig) -> None:
        super().__init__()
        self.config = config
        self.stages = nn.ModuleList(
            SegformerStage(config, index)
            for index in range(len(config.hidden_sizes))
        )

    def forward(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        features = []
        x = pixel_values
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class SegformerMLP(nn.Module):
    """Linear projection of one stage's feature map to the decoder width."""

    def __init__(self, config: SegformerConfig, input_dim: int) -> None:
        super().__init__()
        self.proj = Linear(input_dim, config.decoder_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(map_to_tokens(x))


class SegformerDecodeHead(nn.Module):
    """All-MLP head: project, upsample to 1/4 scale, fuse, classify."""

    def __init__(self, config: SegformerConfig) -> None:
        super().__init__()
        num_stages = len(config.hidden_sizes)
        self.linear_projections = nn.ModuleList(
            SegformerMLP(config, size) for size in config.hidden_sizes
        )
        self.linear_fuse = Conv2d(
            config.decoder_hidden_size * num_stages,
            config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = BatchNorm2d(config.decoder_hidden_size)
        self.activation = ReLU()
        self.dropout = Dropout(CLASSIFIER_DROPOUT)
        self.classifier = Conv2d(
            config.decoder_hidden_size, config.num_labels, kernel_size=1
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        resize = Interpolate(
            size=tuple(features[0].shape[-2:]), mode="bilinear"
        )
        upsampled = []
        for feature, projection in zip(
            features, self.linear_projections, strict=True
        ):
            _, _, height, width = feature.shape
            projected = tokens_to_map(projection(feature), height, width)
            upsampled.append(resize(projected))
        # transformers concatenates the deepest stage first.
        x = self.linear_fuse(torch.cat(upsampled[::-1], dim=1))
        x = self.dropout(self.activation(self.batch_norm(x)))
        return self.classifier(x)


class SegformerForSemanticSegmentation(nn.Module):
    """MiT encoder with the all-MLP decode head; logits at 1/4 resolution."""

    def __init__(self, config: SegformerConfig) -> None:
        super().__init__()
        self.config = config
        self.segformer = SegformerModel(config)
        self.decode_head = SegformerDecodeHead(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.decode_head(self.segformer(pixel_values))


def segformer(key: str) -> SegformerForSemanticSegmentation:
    if key not in SEGFORMER_CONFIGS:
        known = ", ".join(SEGFORMER_CONFIGS)
        raise KeyError(f"unknown SegFormer size {key!r}; known sizes: {known}")
    return SegformerForSemanticSegmentation(SEGFORMER_CONFIGS[key])
