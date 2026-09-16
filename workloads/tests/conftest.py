"""Fixtures shared by every test module."""

from collections.abc import Iterator

import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic_seed() -> Iterator[None]:
    """Seed the global RNG so each test sees the same random tensors."""
    torch.manual_seed(0)
    yield
