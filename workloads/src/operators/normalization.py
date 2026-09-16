"""Root-mean-square layer normalization."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Scale activations by their root mean square, then apply a gain."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Allocate the per-channel gain.

        Args:
            dim: Width of the trailing axis that is normalized.
            eps: Constant added to the variance for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the trailing axis of ``x``.

        The reduction runs in float32 regardless of the input dtype, matching
        the reference Qwen3 implementation.

        Args:
            x: Tensor shaped ``(..., dim)``.

        Returns:
            Tensor of the same shape and dtype as ``x``.
        """
        dtype = x.dtype
        promoted = x.to(torch.float32)
        variance = promoted.pow(2).mean(-1, keepdim=True)
        normalized = promoted * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(dtype)
