"""Token embedding lookup table."""

import torch
from torch import nn


class Embedding(nn.Module):
    """Map integer ids to rows of a learned weight matrix."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        """Allocate the embedding table.

        Args:
            num_embeddings: Number of distinct ids, i.e. the vocabulary size.
            embedding_dim: Width of each embedding row.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Gather the rows named by ``ids``.

        Args:
            ids: Integer tensor of any shape holding row indices.

        Returns:
            Tensor shaped ``ids.shape + (embedding_dim,)``.
        """
        return self.weight[ids]
