import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        promoted = x.to(torch.float32)
        variance = promoted.pow(2).mean(-1, keepdim=True)
        normalized = promoted * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(dtype)
