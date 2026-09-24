"""Fixtures shared by every test module."""

from collections.abc import Iterator

import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic_seed() -> Iterator[None]:
    torch.manual_seed(0)
    yield
