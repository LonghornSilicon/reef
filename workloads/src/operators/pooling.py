import math

import torch
from torch import nn


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int | None = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(2, self.kernel_size, self.stride).unfold(
            3, self.kernel_size, self.stride
        )
        return patches.amax(dim=(-2, -1))


class AdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size: tuple[int, int]) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        out_h, out_w = self.output_size
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
