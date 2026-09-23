"""Normalization operators against their torch and transformers references."""

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from tests.common import DTYPES, assert_matches
from workloads.operators.normalization import BatchNorm2d, RMSNorm

pytestmark = pytest.mark.unit

SHAPES = [(4, 32), (2, 5, 32), (2, 4, 6, 32)]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_rmsnorm_matches_reference(
    shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    ours = RMSNorm(32, eps=1e-6)
    ours.weight.data.normal_()
    reference = Qwen3RMSNorm(32, eps=1e-6)
    reference.weight.data.copy_(ours.weight.data)
    ours, reference = ours.to(dtype), reference.to(dtype)

    x = torch.randn(*shape, dtype=dtype)
    assert_matches(ours(x), reference(x), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_preserves_dtype(dtype: torch.dtype) -> None:
    layer = RMSNorm(32).to(dtype)
    assert layer(torch.randn(2, 32, dtype=dtype)).dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_batchnorm2d_eval_matches_reference(dtype: torch.dtype) -> None:
    ours = BatchNorm2d(8)
    reference = torch.nn.BatchNorm2d(8)
    # Off the mean 0 / variance 1 defaults, where mistakes hide.
    reference.weight.data.normal_()
    reference.bias.data.normal_()
    reference.running_mean.normal_()
    reference.running_var.uniform_(0.5, 2.0)
    ours.load_state_dict(reference.state_dict(), strict=True)
    ours, reference = ours.eval().to(dtype), reference.eval().to(dtype)

    x = torch.randn(4, 8, 5, 5, dtype=dtype)
    assert_matches(ours(x), reference(x), dtype)


def test_batchnorm2d_train_matches_reference() -> None:
    ours, reference = BatchNorm2d(8).train(), torch.nn.BatchNorm2d(8).train()
    reference.weight.data.normal_()
    reference.bias.data.normal_()
    ours.load_state_dict(reference.state_dict(), strict=True)

    x = torch.randn(4, 8, 5, 5)
    torch.testing.assert_close(ours(x), reference(x))


def test_batchnorm2d_running_statistics_track_reference() -> None:
    ours, reference = BatchNorm2d(8).train(), torch.nn.BatchNorm2d(8).train()

    for _ in range(3):
        x = torch.randn(4, 8, 5, 5)
        ours(x)
        reference(x)

    torch.testing.assert_close(ours.running_mean, reference.running_mean)
    torch.testing.assert_close(ours.running_var, reference.running_var)
    assert torch.equal(ours.num_batches_tracked, reference.num_batches_tracked)


def test_batchnorm2d_state_dict_keys_match_reference() -> None:
    ours = BatchNorm2d(8)
    reference = torch.nn.BatchNorm2d(8)
    assert set(ours.state_dict()) == set(reference.state_dict())
