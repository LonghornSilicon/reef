import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        exponent = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        self.register_buffer(
            "inv_freq", 1.0 / (theta**exponent), persistent=False
        )

    def forward(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        angles = positions.to(torch.float32)[:, None] * self.inv_freq[None, :]
        # Split-half layout: both halves share the same frequency ordering.
        doubled = torch.cat((angles, angles), dim=-1)
        return doubled.cos().to(dtype), doubled.sin().to(dtype)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def apply_rotary(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        return x * cos + self.rotate_half(x) * sin
