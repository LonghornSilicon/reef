"""GroupedQueryAttention against ``F.scaled_dot_product_attention``."""

import pytest
import torch
import torch.nn.functional as F

from operators.attention import GroupedQueryAttention
from tests.common import DTYPES, assert_matches

pytestmark = pytest.mark.unit

HEAD_DIM = 16

# (num_heads, num_kv_heads, query_len, key_len)
CASES = [
    (4, 2, 6, 6),  # Grouped, full prefill.
    (4, 4, 6, 6),  # Degenerate group size of one: plain multi-head.
    (8, 1, 5, 5),  # Multi-query.
    (4, 2, 1, 9),  # Single cached decode step.
    (4, 2, 3, 9),  # Chunked prefill against a populated cache.
    (4, 2, 1, 1),  # Single token, empty cache.
]


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    repeats: int,
) -> torch.Tensor:
    """Run causal SDPA with the key/value heads repeated across their group.

    ``is_causal=True`` is only usable when the query and key lengths match:
    PyTorch aligns that mask to the top-left corner, whereas cached decoding
    needs the queries to sit at the *end* of the key axis. For the ragged
    cases the equivalent bottom-right mask is spelled out instead.

    Args:
        query: Tensor shaped ``(batch, num_heads, query_len, head_dim)``.
        key: Tensor shaped ``(batch, num_kv_heads, key_len, head_dim)``.
        value: Tensor shaped ``(batch, num_kv_heads, key_len, head_dim)``.
        repeats: Query heads per key/value head.

    Returns:
        Tensor shaped ``(batch, num_heads, query_len, head_dim)``.
    """
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)
    query_len, key_len = query.shape[2], key.shape[2]
    if query_len == key_len:
        return F.scaled_dot_product_attention(query, key, value, is_causal=True)
    mask = torch.ones(query_len, key_len, dtype=torch.bool).tril(
        diagonal=key_len - query_len
    )
    return F.scaled_dot_product_attention(query, key, value, attn_mask=mask)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("num_heads", "num_kv_heads", "query_len", "key_len"), CASES
)
def test_grouped_query_attention_matches_reference(
    num_heads: int,
    num_kv_heads: int,
    query_len: int,
    key_len: int,
    dtype: torch.dtype,
) -> None:
    """Causal GQA matches SDPA, including when ``query_len < key_len``."""
    query = torch.randn(2, num_heads, query_len, HEAD_DIM, dtype=dtype)
    key = torch.randn(2, num_kv_heads, key_len, HEAD_DIM, dtype=dtype)
    value = torch.randn(2, num_kv_heads, key_len, HEAD_DIM, dtype=dtype)

    attention = GroupedQueryAttention(num_heads, num_kv_heads, HEAD_DIM)
    expected = reference_attention(query, key, value, num_heads // num_kv_heads)
    assert_matches(attention(query, key, value), expected, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_non_causal_attention_matches_reference(dtype: torch.dtype) -> None:
    """With ``causal=False`` every query sees every key."""
    query = torch.randn(2, 4, 5, HEAD_DIM, dtype=dtype)
    key = torch.randn(2, 2, 5, HEAD_DIM, dtype=dtype)
    value = torch.randn(2, 2, 5, HEAD_DIM, dtype=dtype)

    attention = GroupedQueryAttention(4, 2, HEAD_DIM)
    expected = F.scaled_dot_product_attention(
        query,
        key.repeat_interleave(2, dim=1),
        value.repeat_interleave(2, dim=1),
    )
    actual = attention(query, key, value, causal=False)
    assert_matches(actual, expected, dtype)


def test_expand_kv_repeats_each_head_within_its_group() -> None:
    """Key/value heads are repeated in the order the query heads expect."""
    attention = GroupedQueryAttention(6, 3, HEAD_DIM)
    key = torch.randn(1, 3, 4, HEAD_DIM)
    expanded = attention.expand_kv(key)
    assert expanded.shape == (1, 6, 4, HEAD_DIM)
    torch.testing.assert_close(expanded, key.repeat_interleave(2, dim=1))


def test_causal_mask_is_bottom_right_aligned() -> None:
    """A decode step attends to the whole cache plus itself."""
    attention = GroupedQueryAttention(4, 2, HEAD_DIM)
    mask = attention.causal_mask(1, 5, torch.device("cpu"))
    assert not mask.any()

    mask = attention.causal_mask(3, 5, torch.device("cpu"))
    expected = ~torch.ones(3, 5, dtype=torch.bool).tril(diagonal=2)
    torch.testing.assert_close(mask, expected)


# (num_heads, num_kv_heads, query_len, key_len, window)
SLIDING_CASES = [
    (4, 2, 8, 8, 4),  # Prefill where the window bites mid-sequence.
    (4, 2, 8, 8, 1),  # Degenerate window: attend to self only.
    (4, 2, 8, 8, 16),  # Window wider than the sequence: plain causal.
    (4, 2, 1, 12, 4),  # Cached decode step near the end of the window.
    (4, 2, 3, 12, 5),  # Chunked prefill against a populated cache.
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("num_heads", "num_kv_heads", "query_len", "key_len", "window"),
    SLIDING_CASES,
)
def test_sliding_window_matches_reference(
    num_heads: int,
    num_kv_heads: int,
    query_len: int,
    key_len: int,
    window: int,
    dtype: torch.dtype,
) -> None:
    """A windowed layer matches SDPA given the same explicit mask."""
    torch.manual_seed(0)
    shape = (2, num_kv_heads, key_len, HEAD_DIM)
    query = torch.randn(2, num_heads, query_len, HEAD_DIM, dtype=dtype)
    key, value = (
        torch.randn(*shape, dtype=dtype),
        torch.randn(*shape, dtype=dtype),
    )
    layer = GroupedQueryAttention(
        num_heads, num_kv_heads, HEAD_DIM, sliding_window=window
    )

    # Queries occupy the final query_len positions of the key axis.
    query_pos = torch.arange(key_len - query_len, key_len)[:, None]
    key_pos = torch.arange(key_len)[None, :]
    allowed = (key_pos <= query_pos) & (key_pos > query_pos - window)
    repeats = num_heads // num_kv_heads
    expected = F.scaled_dot_product_attention(
        query,
        key.repeat_interleave(repeats, dim=1),
        value.repeat_interleave(repeats, dim=1),
        attn_mask=allowed[None, None],
    )
    assert_matches(layer(query, key, value), expected, dtype)


def test_sliding_window_actually_excludes_distant_keys() -> None:
    """Changing a key outside the window cannot change the output.

    Without this, a window that is silently ignored would still match the
    reference above, because both sides would then be plain causal.
    """
    window = 3
    layer = GroupedQueryAttention(4, 2, HEAD_DIM, sliding_window=window)
    query = torch.randn(1, 4, 8, HEAD_DIM)
    key, value = torch.randn(1, 2, 8, HEAD_DIM), torch.randn(1, 2, 8, HEAD_DIM)

    baseline = layer(query, key, value)
    # Position 0 is outside the window of the final query, which can only see
    # positions 5, 6 and 7.
    value[:, :, 0] += 100.0
    perturbed = layer(query, key, value)

    torch.testing.assert_close(baseline[:, :, -1], perturbed[:, :, -1])
    assert not torch.allclose(baseline[:, :, 0], perturbed[:, :, 0])
