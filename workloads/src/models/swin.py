"""Swin Transformer V1 built from the operator library."""

import torch
from torch import nn
from torch.nn import functional as F

from configs.swin import SWIN_CONFIGS, SwinConfig
from operators.activation import ErfGELU
from operators.attention import GroupedQueryAttention
from operators.convolution import Conv2d
from operators.dropout import Dropout
from operators.linear import Linear
from operators.normalization import LayerNorm
from operators.pooling import AdaptiveAvgPool2d


class Permute(nn.Module):
    """Parameter-free axis reordering, as torchvision.ops.Permute."""

    def __init__(self, dims: tuple[int, ...]) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """View ``(B, H, W, C)`` as ``(B * windows, window_size**2, C)``."""
    batch, height, width, channels = x.shape
    x = x.view(
        batch,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size**2, channels)


def window_reverse(
    x: torch.Tensor, window_size: int, height: int, width: int
) -> torch.Tensor:
    channels = x.shape[-1]
    x = x.view(
        -1,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        channels,
    )
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, height, width, channels)


def relative_position_index(window_size: int) -> torch.Tensor:
    axis = torch.arange(window_size)
    coords = torch.stack(torch.meshgrid(axis, axis, indexing="ij")).flatten(1)
    relative = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0)
    relative[:, :, 0] += window_size - 1
    relative[:, :, 1] += window_size - 1
    relative[:, :, 0] *= 2 * window_size - 1
    return relative.sum(-1).flatten()


def shift_mask(
    height: int,
    width: int,
    window_size: int,
    shift_h: int,
    shift_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Return ``(windows, N, N)``: -100 between regions torch.roll wrapped."""
    regions = torch.zeros(height, width, device=device)
    h_slices = ((0, -window_size), (-window_size, -shift_h), (-shift_h, None))
    w_slices = ((0, -window_size), (-window_size, -shift_w), (-shift_w, None))
    count = 0
    for top, bottom in h_slices:
        for left, right in w_slices:
            regions[top:bottom, left:right] = count
            count += 1
    regions = window_partition(regions[None, :, :, None], window_size)[..., 0]
    different = regions[:, None, :] != regions[:, :, None]
    # torchvision fills -100, not -inf, so the softmax still sees them.
    return torch.where(different, -100.0, 0.0)


class ShiftedWindowAttention(nn.Module):
    """Windowed self-attention with relative position bias and cyclic shift."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        shift_size: int,
        num_heads: int,
        qkv_bias: bool,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.qkv = Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = Linear(dim, dim)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        self.register_buffer(
            "relative_position_index", relative_position_index(window_size)
        )
        self.attention = GroupedQueryAttention(
            num_heads, num_heads, dim // num_heads
        )

    def relative_position_bias(self) -> torch.Tensor:
        tokens = self.window_size**2
        table = self.relative_position_bias_table[self.relative_position_index]
        return table.view(tokens, tokens, -1).permute(2, 0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        size = self.window_size
        pad_w = (size - width % size) % size
        pad_h = (size - height % size) % size
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        padded_h, padded_w = x.shape[1:3]
        # One window covering the whole axis leaves nothing to shift.
        shift_h = 0 if size >= padded_h else self.shift_size
        shift_w = 0 if size >= padded_w else self.shift_size
        shifted = bool(shift_h or shift_w)
        if shifted:
            x = torch.roll(x, shifts=(-shift_h, -shift_w), dims=(1, 2))
        windows = window_partition(x, size)
        tokens = size**2
        qkv = self.qkv(windows).reshape(
            -1, tokens, 3, self.num_heads, channels // self.num_heads
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        bias = self.relative_position_bias()
        if shifted:
            mask = shift_mask(
                padded_h, padded_w, size, shift_h, shift_w, x.device
            )
            bias = (bias[None] + mask[:, None]).expand(batch, -1, -1, -1, -1)
            bias = bias.reshape(-1, self.num_heads, tokens, tokens)
        out = self.attention(query, key, value, causal=False, bias=bias)
        out = self.proj(out.transpose(1, 2).reshape(-1, tokens, channels))
        x = window_reverse(out, size, padded_h, padded_w)
        if shifted:
            x = torch.roll(x, shifts=(shift_h, shift_w), dims=(1, 2))
        return x[:, :height, :width]


class SwinTransformerBlock(nn.Module):
    """Pre-norm block: shifted-window attention, then a GELU MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        shift_size: int,
        config: SwinConfig,
    ) -> None:
        super().__init__()
        eps = config.layer_norm_eps
        self.norm1 = LayerNorm(dim, eps=eps)
        self.attn = ShiftedWindowAttention(
            dim, config.window_size, shift_size, num_heads, config.qkv_bias
        )
        self.norm2 = LayerNorm(dim, eps=eps)
        hidden = int(dim * config.mlp_ratio)
        self.mlp = nn.Sequential(
            Linear(dim, hidden),
            ErfGELU(),
            Dropout(0.0),
            Linear(hidden, dim),
            Dropout(0.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    """Concatenate each 2x2 neighbourhood, normalize, and project 4C to 2C."""

    def __init__(self, dim: int, norm_eps: float) -> None:
        super().__init__()
        self.reduction = Linear(4 * dim, 2 * dim, bias=False)
        self.norm = LayerNorm(4 * dim, eps=norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[1:3]
        x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2))
        x = torch.cat(
            (
                x[:, 0::2, 0::2],
                x[:, 1::2, 0::2],
                x[:, 0::2, 1::2],
                x[:, 1::2, 1::2],
            ),
            dim=-1,
        )
        return self.reduction(self.norm(x))


class SwinTransformer(nn.Module):
    """Hierarchical window-attention classifier matching torchvision."""

    def __init__(self, config: SwinConfig) -> None:
        super().__init__()
        self.config = config
        eps = config.layer_norm_eps
        layers: list[nn.Module] = [
            nn.Sequential(
                Conv2d(
                    3,
                    config.embed_dim,
                    config.patch_size,
                    stride=config.patch_size,
                ),
                Permute((0, 2, 3, 1)),
                LayerNorm(config.embed_dim, eps=eps),
            )
        ]
        stages = list(zip(config.depths, config.num_heads, strict=True))
        for index, (depth, heads) in enumerate(stages):
            dim = config.embed_dim * 2**index
            layers.append(
                nn.Sequential(
                    *(
                        SwinTransformerBlock(
                            dim,
                            heads,
                            0 if i % 2 == 0 else config.window_size // 2,
                            config,
                        )
                        for i in range(depth)
                    )
                )
            )
            if index < len(stages) - 1:
                layers.append(PatchMerging(dim, eps))
        self.features = nn.Sequential(*layers)
        num_features = config.embed_dim * 2 ** (len(stages) - 1)
        self.norm = LayerNorm(num_features, eps=eps)
        self.permute = Permute((0, 3, 1, 2))
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.head = Linear(num_features, config.num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.permute(self.norm(self.features(x)))
        return self.head(self.flatten(self.avgpool(x)))


def swin(key: str) -> SwinTransformer:
    if key not in SWIN_CONFIGS:
        known = ", ".join(SWIN_CONFIGS)
        raise KeyError(f"unknown Swin size {key!r}; known sizes: {known}")
    return SwinTransformer(SWIN_CONFIGS[key])
