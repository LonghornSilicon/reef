"""Spatial pooling operators."""

import math

import torch
from torch import nn


class MaxPool2d(nn.Module):
    """Square-window max pooling."""

    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
    ) -> None:
        """Store the window geometry.

        Args:
            kernel_size: Height and width of the pooling window.
            stride: Step between windows; defaults to ``kernel_size``.
            padding: Implicit border added on every side of the input.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        """Surround ``x`` with a border that can never win a maximum.

        The border is negative infinity rather than zero, so a window that
        overhangs the edge reduces to the largest real value it covers. This
        matches ``F.max_pool2d``, which treats padding as ``-inf`` rather than
        as data.

        Args:
            x: Tensor shaped ``(batch, channels, height, width)``.

        Returns:
            Padded tensor, or ``x`` itself when padding is zero.
        """
        if self.padding == 0:
            return x
        batch, channels, height, width = x.shape
        size = (
            batch,
            channels,
            height + 2 * self.padding,
            width + 2 * self.padding,
        )
        padded = torch.full(size, float("-inf"), dtype=x.dtype, device=x.device)
        row_low, row_high = self.padding, self.padding + height
        col_low, col_high = self.padding, self.padding + width
        padded[:, :, row_low:row_high, col_low:col_high] = x
        return padded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Take the maximum over each window of ``x``.

        Args:
            x: Tensor shaped ``(batch, channels, height, width)``.

        Returns:
            Tensor shaped ``(batch, channels, out_h, out_w)``.
        """
        padded = self.pad(x)
        patches = padded.unfold(2, self.kernel_size, self.stride).unfold(
            3, self.kernel_size, self.stride
        )
        return patches.amax(dim=(-2, -1))


class AdaptiveAvgPool2d(nn.Module):
    """Average pooling onto a fixed output grid of any size."""

    def __init__(self, output_size: tuple[int, int]) -> None:
        """Store the target grid size.

        Args:
            output_size: Desired ``(out_h, out_w)`` of the result.
        """
        super().__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average ``x`` over the window that each output cell covers.

        Window bounds follow PyTorch's convention, so input sizes that are not
        divisible by the output size yield overlapping windows. When they do
        divide evenly the windows tile the input, which the strided fast path
        below handles in one vectorized reduction.

        Args:
            x: Tensor shaped ``(batch, channels, height, width)``.

        Returns:
            Tensor shaped ``(batch, channels) + self.output_size``.
        """
        batch, channels, height, width = x.shape
        out_h, out_w = self.output_size
        if height % out_h == 0 and width % out_w == 0:
            window_h, window_w = height // out_h, width // out_w
            patches = x.unfold(2, window_h, window_h).unfold(
                3, window_w, window_w
            )
            return patches.mean(dim=(-2, -1))
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
