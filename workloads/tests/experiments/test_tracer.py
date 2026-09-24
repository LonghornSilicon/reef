"""Operator records, GEMM shapes and traffic from the meta-device tracer."""

import pytest
import torch

from workloads.experiments import tracer
from workloads.operators.attention import GroupedQueryAttention
from workloads.operators.linear import Linear

pytestmark = pytest.mark.unit


def only(records: list[tracer.Record], operator: str) -> list[tracer.Record]:
    return [record for record in records if record.operator == operator]


def test_linear_record_counts_macs_elements_and_one_gemm() -> None:
    with torch.device("meta"):
        module = Linear(8, 16)
        x = torch.zeros(2, 3, 8)
    (record,) = tracer.trace(module, x)

    assert record.gemms == [(6, 16, 8, 1)]
    assert record.macs == 6 * 8 * 16 + 6 * 16
    assert record.input_numel == 48
    assert record.weight_numel == 16 * 8 + 16
    assert record.output_numel == 96
    assert record.numel == record.fused_numel


def test_attention_records_two_gemms_and_materialized_scores() -> None:
    with torch.device("meta"):
        module = GroupedQueryAttention(4, 2, 16)
        query = torch.zeros(1, 4, 6, 16)
        key = torch.zeros(1, 2, 6, 16)
    records = tracer.trace(module, query, key, key)
    (attention,) = only(records, "GroupedQueryAttention")
    (softmax,) = only(records, "Softmax")

    assert attention.gemms == [(6, 6, 16, 4), (6, 16, 6, 4)]
    assert attention.intermediate_numel == 2 * 4 * 6 * 6
    assert attention.input_numel == query.numel() + 2 * key.numel()
    assert attention.fused_numel == attention.numel - 2 * 4 * 6 * 6
    assert softmax.numel == 2 * 4 * 6 * 6
    assert softmax.fused_numel == 0


def test_decode_gemms_have_batch_rows_and_context_plus_one_keys() -> None:
    records = tracer.decode(
        "gpt_neo", "TinyStories-Instruct-8M", batch=3, context=10
    )

    for record in only(records, "Linear"):
        assert record.gemms[0][0] == 3
    for record in only(records, "GroupedQueryAttention"):
        assert record.gemms[0] == (1, 11, 16, 3 * 16)
        assert record.input_numel == 3 * 16 * (16 + 2 * 16 * 11)


def test_prefill_rejects_positions_the_model_cannot_represent() -> None:
    key, limit = "TinyStories-Instruct-8M", 2048
    tracer.prefill("gpt_neo", key, batch=1, length=limit)
    with pytest.raises(AssertionError):
        tracer.prefill("gpt_neo", key, batch=1, length=limit + 1)
    with pytest.raises(AssertionError):
        tracer.decode("gpt_neo", key, batch=1, context=limit)
