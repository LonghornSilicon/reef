"""Position-derived encodings and attention biases without a token table."""

import math

import torch
from torch import nn

from operators.embedding import Embedding


class SinePositionEmbedding2d(nn.Module):
    """DETR's sine encoding of normalized pixel coordinates."""

    def __init__(
        self,
        num_pos_feats: int,
        temperature: float = 10000.0,
        normalize: bool = True,
        scale: float = 2 * math.pi,
        offset: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        # DETR normalizes cumsum / last; Deformable DETR (cumsum - 0.5) / last.
        self.offset = offset
        self.eps = eps

    def encode(self, embed: torch.Tensor) -> torch.Tensor:
        exponent = torch.arange(self.num_pos_feats, device=embed.device) // 2
        dim_t = self.temperature ** (2 * exponent / self.num_pos_feats)
        angles = embed[..., None] / dim_t
        # Even features take the sine, odd the cosine, interleaved.
        paired = torch.stack(
            (angles[..., 0::2].sin(), angles[..., 1::2].cos()), dim=-1
        )
        return paired.flatten(-2)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        # mask: (batch, height, width), true where the pixel is real.
        # Returns (batch, 2 * num_pos_feats, height, width).
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_last = y_embed[:, -1:, :] + self.eps
            x_last = x_embed[:, :, -1:] + self.eps
            y_embed = (y_embed - self.offset) / y_last * self.scale
            x_embed = (x_embed - self.offset) / x_last * self.scale
        pos = torch.cat((self.encode(y_embed), self.encode(x_embed)), dim=-1)
        return pos.permute(0, 3, 1, 2)


class SinCosPositionEmbedding2d(nn.Module):
    """MAE's fixed sine-cosine table over an integer grid."""

    def __init__(
        self,
        embed_dim: int,
        grid_size: tuple[int, int],
        temperature: float = 10000.0,
        height_first: bool = True,
    ) -> None:
        super().__init__()
        height, width = grid_size
        pos_dim = embed_dim // 4
        exponent = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / temperature**exponent
        rows = torch.arange(height, dtype=torch.float32)
        cols = torch.arange(width, dtype=torch.float32)
        rows = rows[:, None].expand(height, width).reshape(-1)
        cols = cols[None, :].expand(height, width).reshape(-1)
        # HF MAE lays out [sin_h, cos_h, sin_w, cos_w].
        axes = (rows, cols) if height_first else (cols, rows)
        parts = []
        for coordinate in axes:
            angles = coordinate[:, None] * omega[None, :]
            parts += [angles.sin(), angles.cos()]
        self.register_buffer(
            "embedding", torch.cat(parts, dim=1), persistent=False
        )

    def forward(self, dtype: torch.dtype) -> torch.Tensor:
        """Return the ``(height * width, embed_dim)`` table."""
        return self.embedding.to(dtype)


class RelativePositionBias(nn.Module):
    """T5's learned per-head bias over log-bucketed relative positions."""

    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.relative_attention_bias = Embedding(num_buckets, num_heads)

    def bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        num_buckets = self.num_buckets
        if self.bidirectional:
            # Half the buckets for each sign.
            num_buckets //= 2
            buckets = (relative_position > 0).long() * num_buckets
            relative_position = relative_position.abs()
        else:
            buckets = torch.zeros_like(relative_position)
            relative_position = -relative_position.clamp(max=0)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        # Beyond max_exact the buckets widen logarithmically to max_distance.
        scaled = torch.log(relative_position.float() / max_exact)
        scaled = scaled / math.log(self.max_distance / max_exact)
        large = max_exact + (scaled * (num_buckets - max_exact)).long()
        large = large.clamp(max=num_buckets - 1)
        return buckets + torch.where(is_small, relative_position, large)

    def forward(
        self, query_len: int, key_len: int, device: torch.device
    ) -> torch.Tensor:
        """Return a ``(1, num_heads, query_len, key_len)`` additive bias."""
        # Queries occupy the final query_len positions of the key axis.
        query_pos = torch.arange(key_len - query_len, key_len, device=device)
        key_pos = torch.arange(key_len, device=device)
        buckets = self.bucket(key_pos[None, :] - query_pos[:, None])
        return self.relative_attention_bias(buckets).permute(2, 0, 1)[None]
