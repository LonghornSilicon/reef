import torch
from torch import nn


class Dropout(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = (torch.rand_like(x) >= self.p).to(x.dtype)
        return x * keep / (1.0 - self.p)
