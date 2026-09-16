import torch
from torch import nn


class ReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, torch.zeros_like(x))


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (1.0 / (1.0 + torch.exp(-x)))


class Softmax(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shifted = x - x.amax(dim=self.dim, keepdim=True)
        exponentiated = torch.exp(shifted)
        return exponentiated / exponentiated.sum(dim=self.dim, keepdim=True)
