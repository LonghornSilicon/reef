"""Operator records, GEMM shapes and traffic from the meta-device tracer."""

import csv
from pathlib import Path

import pytest
import torch

from experiments import tracer
from models.gpt2 import Conv1D
from operators.attention import GroupedQueryAttention
from operators.convolution import Conv2d
from operators.linear import Linear

pytestmark = pytest.mark.unit

RESULTS = Path(__file__).resolve().parents[2] / "results" / "operator_macs"


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


def test_conv1d_gemm_comes_from_activation_shapes() -> None:
    with torch.device("meta"):
        module = Conv1D(8, 16)
        x = torch.zeros(2, 3, 8)
    (record,) = tracer.trace(module, x)

    assert record.operator == "Linear"
    assert record.gemms == [(6, 16, 8, 1)]


def test_grouped_conv_is_one_gemm_per_group_on_the_unpadded_input() -> None:
    with torch.device("meta"):
        module = Conv2d(8, 16, 3, padding=1, groups=2)
        x = torch.zeros(2, 8, 5, 5)
    (record,) = tracer.trace(module, x)

    assert record.gemms == [(2 * 25, 8, 4 * 9, 2)]
    assert record.input_numel == 2 * 8 * 25
    assert not record.depthwise


def test_depthwise_conv_is_not_a_gemm() -> None:
    with torch.device("meta"):
        module = Conv2d(8, 8, 3, groups=8)
        x = torch.zeros(1, 8, 5, 5)
    (record,) = tracer.trace(module, x)

    assert record.depthwise
    assert record.gemms == []


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
    records = tracer.decode("llama", "SmolLM2-135M", batch=3, context=10)

    for record in only(records, "Linear"):
        assert record.gemms[0][0] == 3
    for record in only(records, "GroupedQueryAttention"):
        assert record.gemms[0] == (1, 11, 64, 3 * 9)
        assert record.input_numel == 3 * 64 * (9 + 2 * 3 * 11)


@pytest.mark.parametrize("family", ["gpt2", "llama", "gpt_neo"])
def test_prefill_rejects_positions_the_model_cannot_represent(
    family: str,
) -> None:
    key = {
        "gpt2": "GPT-2",
        "llama": "SmolLM2-135M",
        "gpt_neo": "TinyStories-Instruct-8M",
    }[family]
    limit = {"gpt2": 1024, "llama": 8192, "gpt_neo": 2048}[family]
    tracer.prefill(family, key, batch=1, length=limit)
    with pytest.raises(AssertionError):
        tracer.prefill(family, key, batch=1, length=limit + 1)
    with pytest.raises(AssertionError):
        tracer.decode(family, key, batch=1, context=limit)


@pytest.mark.parametrize(
    ("family", "key"),
    [
        ("gpt2", "GPT-2"),
        ("llama", "SmolLM2-135M"),
        ("gpt_neo", "TinyStories-Instruct-8M"),
    ],
)
def test_records_reproduce_the_tracked_mac_totals(
    family: str, key: str
) -> None:
    records = tracer.prefill(family, key, batch=1, length=128)
    with open(RESULTS / f"{family}.csv", newline="") as file:
        # A family holds one row group per size, each ending in a Total.
        rows = {
            row["operator"]: row
            for row in csv.DictReader(file)
            if row["model"] == key
        }

    assert sum(record.macs for record in records) == int(rows["Total"]["macs"])
    assert len(records) == int(rows["Total"]["calls"])
