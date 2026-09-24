"""The flops experiment as a script, against its CSV and hand counts."""

import csv
import subprocess
import sys
from collections import defaultdict

import pytest

from workloads.configs.gpt_neo import GPT_NEO_CONFIGS
from workloads.experiments.flops import main as flops
from workloads.experiments.quantization import PRECISIONS

pytestmark = pytest.mark.unit

KEY = "TinyStories-Instruct-8M"
CONFIG = GPT_NEO_CONFIGS[KEY]
OVERHEAD = ("Quantize", "Dequantize")


def linear_weights_and_biases() -> tuple[int, int]:
    """Quantized Linears (everything but lm_head) of one GPT-Neo size."""
    hidden, layers = CONFIG.hidden_size, CONFIG.num_layers
    # q, k, v, out projections, then the 4x-wide MLP in and out.
    weights = layers * (4 * hidden * hidden + 2 * 4 * hidden * hidden)
    # out_proj, c_fc and c_proj carry biases; q, k and v do not.
    biases = layers * (hidden + 4 * hidden + hidden)
    return weights, biases


def total(totals: flops.Totals, operators: tuple[str, ...]) -> int:
    return sum(
        value
        for (operator, _), (_, value) in totals.items()
        if operator in operators
    )


def test_runs_standalone_with_help() -> None:
    subprocess.run(
        [sys.executable, flops.__file__, "--help"],
        check=True,
        capture_output=True,
    )


@pytest.mark.parametrize("phase", flops.PHASES)
@pytest.mark.parametrize("length", [128, 2048])
def test_committed_csv_matches_a_fresh_trace(phase: str, length: int) -> None:
    committed: dict[str, dict] = defaultdict(dict)
    with open(flops.RESULTS / "gpt_neo.csv", newline="") as file:
        for row in csv.DictReader(file):
            if (row["model"], row["phase"], row["tokens"]) == (
                KEY,
                phase,
                str(length),
            ):
                committed[row["precision"]][
                    row["operator"], row["arithmetic"]
                ] = [int(row["calls"]), int(row["flops"])]

    records = flops.trace(KEY, phase, length)
    for precision in PRECISIONS:
        assert dict(flops.count(records, precision)) == committed[precision]


def test_decode_linear_is_constant_and_attention_grows_linearly() -> None:
    weights, biases = linear_weights_and_biases()
    lengths = (2, 64, 2048)
    counts = [
        flops.count(flops.trace(KEY, "decode", length), "fp32")
        for length in lengths
    ]

    for totals in counts:
        assert total(totals, ("Linear",)) == 2 * weights + biases
    per_key = {
        total(totals, ("GroupedQueryAttention",)) / length
        for totals, length in zip(counts, lengths, strict=True)
    }
    assert len(per_key) == 1


@pytest.mark.parametrize("phase", flops.PHASES)
def test_precision_changes_only_the_overhead(phase: str) -> None:
    records = flops.trace(KEY, phase, 64)

    def core(precision: str) -> dict[str, int]:
        summed: dict[str, int] = defaultdict(int)
        for (operator, _), (_, value) in flops.count(
            records, precision
        ).items():
            if operator not in OVERHEAD:
                summed[operator] += value
        return dict(summed)

    for precision in PRECISIONS:
        assert core(precision) == core("fp32"), precision
        if precision in ("fp32", "bf16"):
            assert total(flops.count(records, precision), OVERHEAD) == 0


@pytest.mark.parametrize("phase", flops.PHASES)
def test_quantization_overhead_matches_the_quantized_linears(
    phase: str,
) -> None:
    length = 64
    tokens = length if phase == "prefill" else 1
    weights, _ = linear_weights_and_biases()
    hidden, layers = CONFIG.hidden_size, CONFIG.num_layers
    # Per token: q, k, v and c_fc read hidden-wide inputs, out_proj reads
    # hidden, c_proj reads 4 * hidden; outputs mirror them. lm_head excluded.
    inputs = layers * (4 * hidden + hidden + 4 * hidden)
    outputs = layers * (4 * hidden + 4 * hidden + hidden)
    records = flops.trace(KEY, phase, length)

    int8 = flops.count(records, "int8")
    assert total(int8, OVERHEAD) == tokens * (
        flops.QUANTIZE_FLOPS * inputs + flops.RESCALE_FLOPS * outputs
    )
    # int4 expands every weight once per call, whatever the token count.
    assert total(flops.count(records, "int4"), OVERHEAD) == weights
