"""The kv_cache_size experiment against the cache the model really builds."""

import csv
import subprocess
import sys

import pytest
import torch

from workloads.configs.gpt_neo import GPT_NEO_CONFIGS
from workloads.experiments import tracer
from workloads.experiments.kv_cache_size import main as kv_cache_size

pytestmark = pytest.mark.unit


def test_runs_standalone_with_help() -> None:
    subprocess.run(
        [sys.executable, kv_cache_size.__file__, "--help"],
        check=True,
        capture_output=True,
    )


@pytest.mark.parametrize("key", list(GPT_NEO_CONFIGS))
@pytest.mark.parametrize("length", [1, 300, 2048])
def test_stored_fp32_size_matches_the_model_cache(
    key: str, length: int
) -> None:
    model = tracer.build("gpt_neo", key)
    with torch.no_grad():
        _, cache = model(tracer.tokens(1, length))
    stored = sum(
        tensor.numel() * tensor.element_size()
        for layer in cache
        for tensor in layer
    )

    config = GPT_NEO_CONFIGS[key]
    assert stored == kv_cache_size.kv_bytes(config, length, "fp32", False)


@pytest.mark.parametrize("precision", list(kv_cache_size.PRECISIONS))
def test_needed_matches_stored_up_to_the_window_then_falls_behind(
    precision: str,
) -> None:
    config = GPT_NEO_CONFIGS["TinyStories-Instruct-8M"]

    def size(length: int, windowed: bool) -> int:
        return kv_cache_size.kv_bytes(config, length, precision, windowed)

    window = config.window_size
    assert size(window, True) == size(window, False)
    assert size(window + 1, True) < size(window + 1, False)


def test_committed_csv_matches_the_formula() -> None:
    with open(kv_cache_size.RESULTS / "gpt_neo.csv", newline="") as file:
        rows = list(csv.DictReader(file))

    assert rows
    for row in rows:
        config = GPT_NEO_CONFIGS[row["model"]]
        windowed = row["cache"] == "needed"
        expected = kv_cache_size.kv_bytes(
            config, int(row["tokens"]), row["precision"], windowed
        )
        assert int(row["bytes"]) == expected, row
