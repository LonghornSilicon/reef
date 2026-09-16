"""Grouped-query scaled dot-product attention and its deformable cousin."""

import torch
from torch import nn

from operators.activation import Softmax
from operators.interpolation import GridSample
from operators.linear import Linear
from operators.positional import relative_positions


class GroupedQueryAttention(nn.Module):
    """Attention where several query heads share one key/value head."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float | None = None,
        sliding_window: int | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.repeats = num_heads // num_kv_heads
        self.scale = head_dim**-0.5 if scale is None else scale
        self.sliding_window = sliding_window
        self.softmax = Softmax(dim=-1)

    def expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.repeats == 1:
            return x
        batch, heads, length, dim = x.shape
        expanded = x[:, :, None].expand(batch, heads, self.repeats, length, dim)
        return expanded.reshape(batch, heads * self.repeats, length, dim)

    def causal_mask(
        self, query_len: int, key_len: int, device: torch.device
    ) -> torch.Tensor:
        """Return a ``(query_len, key_len)`` mask, true where disallowed."""
        relative = relative_positions(query_len, key_len, device)
        masked = relative > 0
        if self.sliding_window is not None:
            # The window includes the query's own position.
            masked = masked | (relative <= -self.sliding_window)
        return masked

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = True,
        mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # query: (batch, num_heads, query_len, head_dim)
        # key, value: (batch, num_kv_heads, key_len, head_dim)
        # mask: bool, true where disallowed; bias: additive scores. Both
        # broadcast to (batch, num_heads, query_len, key_len).
        key = self.expand_kv(key)
        value = self.expand_kv(value)
        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        if bias is not None:
            # Add before the fill: finfo.min plus a negative bias overflows
            # to -inf and a fully masked row would softmax to NaN.
            scores = scores + bias
        if causal:
            disallowed = self.causal_mask(
                query.shape[2], key.shape[2], query.device
            )
            mask = disallowed if mask is None else disallowed | mask
        if mask is not None:
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        weights = self.softmax(scores.to(torch.float32)).to(value.dtype)
        return torch.matmul(weights, value)


class MultiScaleDeformableAttention(nn.Module):
    """Deformable DETR attention over sampled points near reference points."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_levels: int,
        num_points: int,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads
        samples = num_heads * num_levels * num_points
        self.sampling_offsets = Linear(embed_dim, samples * 2)
        self.attention_weights = Linear(embed_dim, samples)
        self.value_proj = Linear(embed_dim, embed_dim)
        self.output_proj = Linear(embed_dim, embed_dim)
        self.softmax = Softmax(dim=-1)
        self.grid_sample = GridSample()

    def locations(
        self, offsets: torch.Tensor, reference_points: torch.Tensor
    ) -> torch.Tensor:
        """Return sampling points in [0, 1] image coordinates."""
        reference = reference_points[:, :, None, :, None, :]
        if reference_points.shape[-1] == 2:
            return reference + offsets
        # Box references: offsets are fractions of the box half-size.
        scaled = offsets / self.num_points * reference[..., 2:] * 0.5
        return reference[..., :2] + scaled

    def forward(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: list[tuple[int, int]],
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # query: (batch, num_queries, embed_dim)
        # value: (batch, num_values, embed_dim), every level flattened and
        # concatenated in spatial_shapes order
        # reference_points: (batch, num_queries, num_levels, 2 or 4) in [0, 1]
        # mask: (batch, num_values), true where the value is padding
        batch, num_queries, _ = query.shape
        heads, levels, points = self.num_heads, self.num_levels, self.num_points
        value = self.value_proj(value)
        if mask is not None:
            value = value.masked_fill(mask[..., None], 0.0)
        value = value.reshape(batch, -1, heads, self.head_dim)
        offsets = self.sampling_offsets(query).reshape(
            batch, num_queries, heads, levels, points, 2
        )
        if reference_points.shape[-1] == 2:
            # Pixel offsets become fractions of each level's (width, height).
            extents = [(width, height) for height, width in spatial_shapes]
            normalizer = torch.tensor(
                extents, dtype=offsets.dtype, device=offsets.device
            )
            offsets = offsets / normalizer[None, None, None, :, None, :]
        weights = self.attention_weights(query).reshape(
            batch, num_queries, heads, levels * points
        )
        weights = self.softmax(weights)
        grids = 2 * self.locations(offsets, reference_points) - 1
        # (batch * heads, num_queries, levels, points, 2)
        grids = grids.transpose(1, 2).flatten(0, 1)
        sampled = []
        start = 0
        for level, (height, width) in enumerate(spatial_shapes):
            level_value = value[:, start : start + height * width]
            level_value = level_value.permute(0, 2, 3, 1).reshape(
                batch * heads, self.head_dim, height, width
            )
            sampled.append(self.grid_sample(level_value, grids[:, :, level]))
            start += height * width
        # (batch * heads, head_dim, num_queries, levels * points)
        sampled = torch.stack(sampled, dim=-2).flatten(-2)
        weights = weights.transpose(1, 2).reshape(
            batch * heads, 1, num_queries, levels * points
        )
        out = (sampled * weights).sum(-1)
        out = out.reshape(batch, self.embed_dim, num_queries).transpose(1, 2)
        return self.output_proj(out)
