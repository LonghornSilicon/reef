"""Spatial pooling operators."""

import math

import torch
from torch import nn

from operators.convolution import Padding, expand_padding, pad2d, windows2d


def pooled_size(
    size: int,
    kernel_size: int,
    stride: int,
    low: int,
    high: int,
    ceil_mode: bool,
) -> int:
    span = size + low + high - kernel_size
    out = math.ceil(span / stride) + 1 if ceil_mode else span // stride + 1
    # torch drops a trailing window that would start inside the high padding.
    if ceil_mode and (out - 1) * stride >= size + low:
        out -= 1
    return out


def pad_windows(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: Padding,
    ceil_mode: bool,
    value: float,
) -> tuple[torch.Tensor, int, int]:
    """Pad so that ``unfold`` covers every window torch would pool."""
    left, right, top, bottom = expand_padding(padding)
    height, width = x.shape[2:]
    out_h = pooled_size(height, kernel_size, stride, top, bottom, ceil_mode)
    out_w = pooled_size(width, kernel_size, stride, left, right, ceil_mode)
    extra_h = (out_h - 1) * stride + kernel_size - height - top - bottom
    extra_w = (out_w - 1) * stride + kernel_size - width - left - right
    padded = pad2d(
        x, (left, right + max(extra_w, 0), top, bottom + max(extra_h, 0)), value
    )
    return padded, out_h, out_w


class MaxPool2d(nn.Module):
    """Square-window max pooling."""

    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: Padding = 0,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad with -inf, not zero, as F.max_pool2d does.
        padded, out_h, out_w = pad_windows(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            float("-inf"),
        )
        patches = windows2d(padded, self.kernel_size, self.stride)
        return patches[:, :, :out_h, :out_w].amax(dim=(-2, -1))


class AvgPool2d(nn.Module):
    """Square-window average pooling."""

    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: Padding = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def total(self, x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        patches = windows2d(x, self.kernel_size, self.stride)
        return patches[:, :, :out_h, :out_w].sum(dim=(-2, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded, out_h, out_w = pad_windows(
            x, self.kernel_size, self.stride, self.padding, self.ceil_mode, 0.0
        )
        # The divisor counts the explicit padding only when count_include_pad,
        # and never the ceil_mode overhang, matching F.avg_pool2d.
        counted = pad2d(
            torch.ones_like(x[:1, :1]),
            self.padding,
            1.0 if self.count_include_pad else 0.0,
        )
        overhang_h = padded.shape[2] - counted.shape[2]
        overhang_w = padded.shape[3] - counted.shape[3]
        counted = pad2d(counted, (0, overhang_w, 0, overhang_h), 0.0)
        return self.total(padded, out_h, out_w) / self.total(
            counted, out_h, out_w
        )


class AdaptiveAvgPool2d(nn.Module):
    """Average pooling onto a fixed output grid of any size."""

    def __init__(self, output_size: tuple[int, int]) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        out_h, out_w = self.output_size
        if height % out_h == 0 and width % out_w == 0:
            window = (height // out_h, width // out_w)
            return windows2d(x, window, window).mean(dim=(-2, -1))
        # PyTorch's window bounds: sizes that do not divide evenly yield
        # overlapping windows.
        out = torch.zeros(
            (batch, channels, out_h, out_w), dtype=x.dtype, device=x.device
        )
        for i in range(out_h):
            top = (i * height) // out_h
            bottom = math.ceil((i + 1) * height / out_h)
            for j in range(out_w):
                left = (j * width) // out_w
                right = math.ceil((j + 1) * width / out_w)
                window = x[:, :, top:bottom, left:right]
                out[:, :, i, j] = window.mean(dim=(-2, -1))
        return out
