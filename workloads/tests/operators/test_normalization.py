"""RMSNorm against the reference ``Qwen3RMSNorm`` from transformers."""

import pytest
import torch
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from operators.normalization import BatchNorm2d, GemmaRMSNorm, RMSNorm
from tests.common import DTYPES, assert_matches

pytestmark = pytest.mark.unit

SHAPES = [(4, 32), (2, 5, 32), (2, 4, 6, 32)]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_rmsnorm_matches_reference(
    shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    """RMSNorm matches ``Qwen3RMSNorm`` with the same learned gain."""
    ours = RMSNorm(32, eps=1e-6)
    ours.weight.data.normal_()
    reference = Qwen3RMSNorm(32, eps=1e-6)
    reference.weight.data.copy_(ours.weight.data)
    ours, reference = ours.to(dtype), reference.to(dtype)

    x = torch.randn(*shape, dtype=dtype)
    assert_matches(ours(x), reference(x), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_preserves_dtype(dtype: torch.dtype) -> None:
    """The float32 reduction does not leak into the output dtype."""
    layer = RMSNorm(32).to(dtype)
    assert layer(torch.randn(2, 32, dtype=dtype)).dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_batchnorm2d_eval_matches_reference(dtype: torch.dtype) -> None:
    """BatchNorm2d matches ``nn.BatchNorm2d`` using running statistics."""
    ours = BatchNorm2d(8)
    reference = torch.nn.BatchNorm2d(8)
    # Move the statistics off their defaults so a mistake cannot hide behind
    # mean 0 / variance 1.
    reference.weight.data.normal_()
    reference.bias.data.normal_()
    reference.running_mean.normal_()
    reference.running_var.uniform_(0.5, 2.0)
    ours.load_state_dict(reference.state_dict(), strict=True)
    ours, reference = ours.eval().to(dtype), reference.eval().to(dtype)

    x = torch.randn(4, 8, 5, 5, dtype=dtype)
    assert_matches(ours(x), reference(x), dtype)


def test_batchnorm2d_train_matches_reference() -> None:
    """In train mode both normalize by the same batch statistics."""
    ours, reference = BatchNorm2d(8).train(), torch.nn.BatchNorm2d(8).train()
    reference.weight.data.normal_()
    reference.bias.data.normal_()
    ours.load_state_dict(reference.state_dict(), strict=True)

    x = torch.randn(4, 8, 5, 5)
    torch.testing.assert_close(ours(x), reference(x))


def test_batchnorm2d_running_statistics_track_reference() -> None:
    """The running mean, variance and batch counter update identically.

    PyTorch normalizes by the biased variance but accumulates the unbiased
    one into ``running_var``; this is the test that pins that asymmetry.
    """
    ours, reference = BatchNorm2d(8).train(), torch.nn.BatchNorm2d(8).train()

    for _ in range(3):
        x = torch.randn(4, 8, 5, 5)
        ours(x)
        reference(x)

    torch.testing.assert_close(ours.running_mean, reference.running_mean)
    torch.testing.assert_close(ours.running_var, reference.running_var)
    assert torch.equal(ours.num_batches_tracked, reference.num_batches_tracked)


def test_batchnorm2d_state_dict_keys_match_reference() -> None:
    """Our buffer names match, so checkpoints load without renaming."""
    ours = BatchNorm2d(8)
    reference = torch.nn.BatchNorm2d(8)
    assert set(ours.state_dict()) == set(reference.state_dict())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_gemma_rmsnorm_matches_reference(
    shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    """GemmaRMSNorm matches ``Gemma3RMSNorm`` with the same learned gain."""
    ours = GemmaRMSNorm(32, eps=1e-6)
    ours.weight.data.normal_()
    reference = Gemma3RMSNorm(32, eps=1e-6)
    reference.weight.data.copy_(ours.weight.data)
    ours, reference = ours.to(dtype), reference.to(dtype)

    x = torch.randn(*shape, dtype=dtype)
    assert_matches(ours(x), reference(x), dtype)


def test_gemma_rmsnorm_gain_is_an_offset_from_one() -> None:
    """A zero weight is the identity gain, not a zero gain.

    Gemma stores the gain as ``1 + weight``, so the zero-initialized default
    must pass the normalized values through untouched. Under RMSNorm's
    one-initialized convention this would return zeros.
    """
    layer = GemmaRMSNorm(32)
    torch.testing.assert_close(layer.weight, torch.zeros(32))

    x = torch.randn(2, 32)
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + layer.eps)
    torch.testing.assert_close(layer(x), expected)


def test_gemma_rmsnorm_preserves_dtype() -> None:
    """The float32 reduction does not leak into the output dtype."""
    layer = GemmaRMSNorm(32).to(torch.bfloat16)
    x = torch.randn(2, 32, dtype=torch.bfloat16)
    assert layer(x).dtype == torch.bfloat16
