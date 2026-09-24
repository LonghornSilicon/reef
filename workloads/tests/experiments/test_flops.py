"""The operator_macs experiment as a script and against its committed CSVs."""

import csv
import subprocess
import sys

import pytest

from workloads.experiments import tracer
from workloads.experiments.operator_macs import main as operator_macs

pytestmark = pytest.mark.unit


def test_runs_standalone_with_help() -> None:
    subprocess.run(
        [sys.executable, operator_macs.__file__, "--help"],
        check=True,
        capture_output=True,
    )


def test_records_reproduce_the_tracked_mac_totals() -> None:
    key = "TinyStories-Instruct-8M"
    records = tracer.prefill("gpt_neo", key, 1, operator_macs.SEQ_LEN)
    with open(operator_macs.RESULTS / "gpt_neo.csv", newline="") as file:
        # A family holds one row group per size, each ending in a Total.
        rows = {
            row["operator"]: row
            for row in csv.DictReader(file)
            if row["model"] == key
        }

    assert sum(record.macs for record in records) == int(rows["Total"]["macs"])
    assert len(records) == int(rows["Total"]["calls"])
