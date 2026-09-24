"""The story_quality experiment as a script and against its committed CSVs."""

import csv
import itertools
import subprocess
import sys

import pytest
import torch
from torch import nn

from workloads.configs.gpt_neo import GPT_NEO_CONFIGS, GPTNeoConfig
from workloads.experiments.story_quality import main as story_quality
from workloads.models.gpt_neo import GPTNeoForCausalLM
from workloads.operators.linear import Linear

TINY = GPTNeoConfig(hidden_size=64, num_layers=2, num_heads=4, vocab_size=1000)


def quantized_linears(model: nn.Module) -> list[Linear]:
    return [
        module
        for name, module in model.named_modules()
        if isinstance(module, Linear) and name != "lm_head"
    ]


def assert_on_grid(vectors: torch.Tensor, bits: int) -> None:
    """Each last-axis vector is an integer multiple of its amax / limit."""
    limit = 2 ** (bits - 1) - 1
    steps = vectors / vectors.abs().amax(dim=-1, keepdim=True) * limit
    torch.testing.assert_close(steps, steps.round(), rtol=0, atol=1e-4)


def assert_head_and_source_untouched(
    model: GPTNeoForCausalLM,
    converted: GPTNeoForCausalLM,
    original: dict[str, torch.Tensor],
) -> None:
    head = converted.lm_head.weight
    assert head is converted.transformer.wte.weight
    assert torch.equal(head, original["lm_head.weight"])
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, original[name]), name


@pytest.mark.unit
def test_runs_standalone_with_help() -> None:
    subprocess.run(
        [sys.executable, story_quality.__file__, "--help"],
        check=True,
        capture_output=True,
    )


@pytest.mark.unit
def test_int8_rounds_each_linear_row_and_leaves_the_tied_head() -> None:
    model = GPTNeoForCausalLM(TINY)
    original = {name: t.clone() for name, t in model.state_dict().items()}

    converted = story_quality.convert(model, "int8")

    linears = quantized_linears(converted)
    assert linears
    for module in linears:
        assert_on_grid(module.weight, bits=8)
    assert_head_and_source_untouched(model, converted, original)


@pytest.mark.unit
def test_int4_rounds_each_group_of_32_and_leaves_the_tied_head() -> None:
    model = GPTNeoForCausalLM(TINY)
    original = {name: t.clone() for name, t in model.state_dict().items()}

    converted = story_quality.convert(model, "int4")

    linears = quantized_linears(converted)
    assert linears
    for module in linears:
        groups = module.weight.reshape(
            module.out_features, -1, story_quality.INT4_GROUP
        )
        assert_on_grid(groups, bits=4)
    assert_head_and_source_untouched(model, converted, original)


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    ("key", "precision"),
    list(itertools.product(GPT_NEO_CONFIGS, story_quality.PRECISIONS)),
)
def test_greedy_stories_match_the_committed_results(
    key: str, precision: str
) -> None:
    fp32, tokenizer = story_quality.load(key)
    model = story_quality.convert(fp32, precision)
    for index, (name, prompt) in enumerate(story_quality.PROMPTS, start=1):
        with open(story_quality.csv_path(index, name), newline="") as file:
            rows = csv.DictReader(file)
            stories = {
                (row["model"], row["precision"]): row["story"] for row in rows
            }

        story = story_quality.tell(model, tokenizer, prompt)
        assert story == stories[key, precision], name
