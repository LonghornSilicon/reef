"""Normalization operators."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Scale activations by their root mean square, then apply a gain."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        # float32 regardless of input dtype, matching the reference Qwen3.
        promoted = x.to(torch.float32)
        variance = promoted.pow(2).mean(-1, keepdim=True)
        normalized = promoted * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(dtype)


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


class GroupNorm(nn.Module):
    """Normalize channel groups over every axis after the batch."""

    def __init__(
        self, num_groups: int, num_channels: int, eps: float = 1e-5
    ) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        promoted = x.to(torch.float32).reshape(batch, self.num_groups, -1)
        mean = promoted.mean(-1, keepdim=True)
        variance = promoted.var(-1, correction=0, keepdim=True)
        normalized = (promoted - mean) * torch.rsqrt(variance + self.eps)
        normalized = normalized.reshape(x.shape).to(x.dtype)
        shape = (1, self.num_channels) + (1,) * (x.dim() - 2)
        scaled = normalized * self.weight.reshape(shape)
        return scaled + self.bias.reshape(shape)


class L2Norm(nn.Module):
    """Scale each vector along an axis to unit Euclidean length."""

    def __init__(self, dim: int = -1, eps: float = 1e-12) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.to(torch.float32).pow(2).sum(self.dim, keepdim=True).sqrt()
        # Clamp, not add: F.normalize divides by max(norm, eps).
        return x / norm.clamp(min=self.eps).to(x.dtype)


class BatchNorm2d(nn.Module):
    """Normalize each channel of a feature map by its batch statistics."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer(
            "num_batches_tracked", torch.tensor(0, dtype=torch.long)
        )

    def _batch_statistics(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reduced = (0, 2, 3)
        mean = x.mean(dim=reduced)
        variance = x.var(dim=reduced, correction=0)
        samples = x.numel() // x.shape[1]
        with torch.no_grad():
            self.num_batches_tracked += 1
            # Normalization uses the biased variance but the running estimate
            # takes the unbiased one, as torch.nn.BatchNorm2d does.
            unbiased = variance * samples / (samples - 1)
            keep = 1.0 - self.momentum
            self.running_mean.mul_(keep).add_(self.momentum * mean)
            self.running_var.mul_(keep).add_(self.momentum * unbiased)
        return mean, variance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean, variance = self._batch_statistics(x)
        else:
            mean, variance = self.running_mean, self.running_var
        shape = (1, -1, 1, 1)
        mean = mean.to(x.dtype).reshape(shape)
        variance = variance.to(x.dtype).reshape(shape)
        normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        scaled = normalized * self.weight.reshape(shape)
        return scaled + self.bias.reshape(shape)


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d whose statistics and affine terms are fixed buffers."""

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = (1, -1, 1, 1)
        scale = self.weight * torch.rsqrt(self.running_var + self.eps)
        shift = self.bias - self.running_mean * scale
        scale = scale.reshape(shape).to(x.dtype)
        shift = shift.reshape(shape).to(x.dtype)
        return x * scale + shift


class GemmaRMSNorm(nn.Module):
    """RMS normalization in Gemma's parameterization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        promoted = x.to(torch.float32)
        variance = promoted.pow(2).mean(-1, keepdim=True)
        normalized = promoted * torch.rsqrt(variance + self.eps)
        # Multiply then cast, the reverse of RMSNorm, matching Gemma3RMSNorm.
        scaled = normalized * (1.0 + self.weight.to(torch.float32))
        return scaled.to(x.dtype)
