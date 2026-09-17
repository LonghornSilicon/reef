"""YOLO11 detector built from the operator library."""

import torch
from torch import nn

from configs.yolo11 import YOLO11_CONFIGS, YOLO11Config
from models.yolov8 import (
    SPPF,
    Bottleneck,
    C2f,
    Concat,
    Conv,
    Detect,
    Detector,
    Row,
)
from operators.attention import GroupedQueryAttention
from operators.interpolation import Interpolate


class C3k(nn.Module):
    """CSP stage with three convolutions and a serial bottleneck chain."""

    def __init__(
        self, c1: int, c2: int, n: int, shortcut: bool = True, k: int = 3
    ) -> None:
        super().__init__()
        hidden = c2 // 2
        self.cv1 = Conv(c1, hidden)
        self.cv2 = Conv(c1, hidden)
        self.cv3 = Conv(2 * hidden, c2)
        self.m = nn.Sequential(
            *(
                Bottleneck(hidden, hidden, shortcut, (k, k), e=1.0)
                for _ in range(n)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k2(C2f):
    """C2f whose inner blocks are C3k stages (m/l/x) or bottlenecks (n/s)."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int,
        c3k: bool = False,
        e: float = 0.5,
        shortcut: bool = True,
    ) -> None:
        super().__init__(c1, c2, n, shortcut, e)
        # Unlike C2f's bottlenecks these keep the default 0.5 expansion.
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut)
            if c3k
            else Bottleneck(self.c, self.c, shortcut)
            for _ in range(n)
        )


class Attention(nn.Module):
    """Multi-head self-attention over pixels with a depthwise position term."""

    def __init__(
        self, dim: int, num_heads: int, attn_ratio: float = 0.5
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.qkv = Conv(dim, dim + 2 * self.key_dim * num_heads, act=False)
        self.proj = Conv(dim, dim, act=False)
        self.pe = Conv(dim, dim, 3, groups=dim, act=False)
        self.attention = GroupedQueryAttention(
            num_heads, num_heads, self.head_dim, scale=self.key_dim**-0.5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        qkv = self.qkv(x).reshape(
            batch, self.num_heads, 2 * self.key_dim + self.head_dim, -1
        )
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        out = self.attention(
            q.transpose(-1, -2),
            k.transpose(-1, -2),
            v.transpose(-1, -2),
            causal=False,
        )
        out = out.transpose(-1, -2).reshape(batch, channels, height, width)
        out = out + self.pe(v.reshape(batch, channels, height, width))
        return self.proj(out)


class PSABlock(nn.Module):
    """Attention then a 1x1 feed-forward, each with a residual."""

    def __init__(self, c: int, num_heads: int) -> None:
        super().__init__()
        self.attn = Attention(c, num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2), Conv(c * 2, c, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.ffn(x)


class C2PSA(nn.Module):
    """Split channels, run PSA blocks on one half, concat and project."""

    def __init__(self, c1: int, c2: int, n: int, e: float = 0.5) -> None:
        super().__init__()
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c)
        self.cv2 = Conv(2 * self.c, c1)
        self.m = nn.Sequential(
            *(PSABlock(self.c, max(self.c // 64, 1)) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        return self.cv2(torch.cat((a, self.m(b)), 1))


ROWS: list[Row] = [
    (-1, 1, Conv, [64, 3, 2]),
    (-1, 1, Conv, [128, 3, 2]),
    (-1, 2, C3k2, [256, False, 0.25]),
    (-1, 1, Conv, [256, 3, 2]),
    (-1, 2, C3k2, [512, False, 0.25]),
    (-1, 1, Conv, [512, 3, 2]),
    (-1, 2, C3k2, [512, True]),
    (-1, 1, Conv, [1024, 3, 2]),
    (-1, 2, C3k2, [1024, True]),
    (-1, 1, SPPF, [1024, 5]),
    (-1, 2, C2PSA, [1024]),
    (-1, 1, Interpolate, []),
    ([-1, 6], 1, Concat, []),
    (-1, 2, C3k2, [512, False]),
    (-1, 1, Interpolate, []),
    ([-1, 4], 1, Concat, []),
    (-1, 2, C3k2, [256, False]),
    (-1, 1, Conv, [256, 3, 2]),
    ([-1, 13], 1, Concat, []),
    (-1, 2, C3k2, [512, False]),
    (-1, 1, Conv, [512, 3, 2]),
    ([-1, 10], 1, Concat, []),
    (-1, 2, C3k2, [1024, True]),
    ([16, 19, 22], 1, Detect, []),
]


class YOLO11(Detector):
    """YOLO11 with C3k2 stages, C2PSA attention and the depthwise head."""

    def __init__(self, config: YOLO11Config) -> None:
        rows = [
            (source, n, block, [args[0], args[1] or config.c3k, *args[2:]])
            if block is C3k2
            else (source, n, block, args)
            for source, n, block, args in ROWS
        ]
        super().__init__(
            rows, config, legacy=False, repeated=(C2f, C3k2, C2PSA)
        )


def yolo11(name: str) -> YOLO11:
    if name not in YOLO11_CONFIGS:
        known = ", ".join(YOLO11_CONFIGS)
        raise KeyError(f"unknown YOLO11 size {name!r}; known sizes: {known}")
    return YOLO11(YOLO11_CONFIGS[name])
