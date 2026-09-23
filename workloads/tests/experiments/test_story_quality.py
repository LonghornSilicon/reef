"""The story_quality experiment as a script and against its committed CSVs."""

import csv
import subprocess
import sys

import pytest

from workloads.configs.gpt_neo import GPT_NEO_CONFIGS
from workloads.experiments.story_quality import main as story_quality


@pytest.mark.unit
def test_runs_standalone_with_help() -> None:
    subprocess.run(
        [sys.executable, story_quality.__file__, "--help"],
        check=True,
        capture_output=True,
    )


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("key", list(GPT_NEO_CONFIGS))
def test_greedy_stories_match_the_committed_results(key: str) -> None:
    model, tokenizer = story_quality.load(key)
    for index, (name, prompt) in enumerate(story_quality.PROMPTS, start=1):
        with open(story_quality.csv_path(index, name), newline="") as file:
            rows = csv.DictReader(file)
            expected = {row["model"]: row["story"] for row in rows}[key]

        story = story_quality.tell(model, tokenizer, prompt)
        assert story == expected, name
