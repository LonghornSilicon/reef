"""Normalization operators."""

import torch
from torch import nn


class LayerNorm(nn.Module):
    """Normalize over the last axis, then apply a gain and optional shift."""

    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        promoted = x.to(torch.float32)
        mean = promoted.mean(-1, keepdim=True)
        centred = promoted - mean
        variance = centred.pow(2).mean(-1, keepdim=True)
        normalized = centred * torch.rsqrt(variance + self.eps)
        out = normalized.to(x.dtype) * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out
