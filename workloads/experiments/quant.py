"""Symmetric fake quantization and the storage cost of each precision."""

import math

import torch
from torch import nn

from models.gpt2 import Conv1D
from operators.attention import GroupedQueryAttention
from operators.convolution import Conv2d
from operators.linear import Linear

BITS = {"bf16": 16, "int8": 8, "int4": 4}
# fp16 scale per channel, per group or per (token, head) vector.
SCALE_BYTES = 2
GROUP = 128


def quantize(x: torch.Tensor, amax: torch.Tensor, bits: int) -> torch.Tensor:
    """Round ``x`` to a symmetric ``bits``-wide grid spanning ``±amax``."""
    limit = 2 ** (bits - 1) - 1
    # An all-zero row has amax 0; keep the scale finite so 0 / scale is 0.
    scale = amax.clamp(min=torch.finfo(amax.dtype).tiny) / limit
    return torch.clamp(torch.round(x / scale), -limit, limit) * scale


def quantize_per_channel(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """``weight`` is ``(channels, features)``; one scale per channel."""
    return quantize(weight, weight.abs().amax(dim=1, keepdim=True), bits)


def quantize_group(
    weight: torch.Tensor, bits: int, group: int = GROUP
) -> torch.Tensor:
    """``weight`` is ``(channels, features)``; one scale per ``group``."""
    channels, features = weight.shape
    groups = math.ceil(features / group)
    # Zero padding fills the last partial group without moving its amax.
    padded = torch.zeros(channels, groups * group, dtype=weight.dtype)
    padded[:, :features] = weight
    grouped = padded.reshape(channels, groups, group)
    quantized = quantize(grouped, grouped.abs().amax(dim=2, keepdim=True), bits)
    return quantized.reshape(channels, groups * group)[:, :features]


def quantize_per_token(x: torch.Tensor, bits: int) -> torch.Tensor:
    """One scale per vector along the last axis."""
    return quantize(x, x.abs().amax(dim=-1, keepdim=True), bits)


def channel_view(module: nn.Module) -> torch.Tensor:
    """The weight as ``(output channels, features)``, without copying."""
    if isinstance(module, Conv1D):
        return module.weight.t()
    if isinstance(module, Conv2d):
        return module.weight.reshape(module.out_channels, -1)
    return module.weight


def quantizable(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Weight-bearing matmul modules, tied weights once, no ``lm_head``."""
    seen: set[int] = set()
    modules = []
    for name, module in model.named_modules():
        if not isinstance(module, Linear | Conv1D | Conv2d):
            continue
        if name == "lm_head" or id(module.weight) in seen:
            continue
        seen.add(id(module.weight))
        modules.append((name, module))
    return modules


def weight_bytes(numel: int, channels: int, scheme: str) -> float:
    if scheme == "bf16":
        scales = 0
    elif scheme == "int8":
        scales = channels
    else:
        scales = channels * math.ceil(numel / channels / GROUP)
    return numel * BITS[scheme] / 8 + scales * SCALE_BYTES


def model_weight_bytes(model: nn.Module, scheme: str) -> float:
    """Matmul weights at ``scheme``; everything else stays bf16."""
    quantized: set[int] = set()
    total = 0.0
    for _, module in quantizable(model):
        weight = channel_view(module)
        total += weight_bytes(weight.numel(), weight.shape[0], scheme)
        quantized.add(id(module.weight))
    rest = sum(
        parameter.numel()
        for parameter in model.parameters()
        if id(parameter) not in quantized
    )
    return total + rest * BITS["bf16"] / 8


def attentions(model: nn.Module) -> list[GroupedQueryAttention]:
    return [
        module
        for module in model.modules()
        if isinstance(module, GroupedQueryAttention)
    ]


def kv_elements_per_token(model: nn.Module) -> int:
    return sum(
        2 * module.num_kv_heads * module.head_dim
        for module in attentions(model)
    )


def kv_bytes_per_token(model: nn.Module, scheme: str) -> float:
    vectors = sum(2 * module.num_kv_heads for module in attentions(model))
    scales = 0 if scheme == "bf16" else vectors * SCALE_BYTES
    return kv_elements_per_token(model) * BITS[scheme] / 8 + scales
