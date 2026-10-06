"""Run the full `run` sweep and check its artifacts look reasonable."""

import csv
import json
from pathlib import Path

import pytest

from reef_perf.experiments.run import plot
from reef_perf.experiments.run.run import load_config, run_sweep
from reef_perf.experiments.run.workloads import list_workloads
from reef_perf.sim import MODULE_ROOT

ALL_CONFIG = MODULE_ROOT / "src/reef_perf/experiments/run/configs/all.yaml"


@pytest.mark.experiment
@pytest.mark.slow
def test_all_sweep_collects_every_artifact(
    cpp_build_dir: Path, tmp_path: Path
) -> None:
    """configs/all.yaml produces one sane artifact per workload, plus plots."""
    del cpp_build_dir  # built for its side effect
    config = load_config(ALL_CONFIG)
    assert config.workloads == list_workloads()

    paths = run_sweep(config, results_root=tmp_path)
    sweep_dir = tmp_path / config.name
    assert len(paths) == len(config.workloads) * len(config.variants)

    for path in paths:
        artifact = json.loads(path.read_text())
        result = artifact["result"]
        name = artifact["workload"]
        assert result["instructions"] > 0, name
        assert result["instructions"] == result["functional_instructions"], name
        assert result["cycles"] >= result["instructions"] / 4, name
        assert 0.0 < result["ipc"] <= 4.0, name

    with (sweep_dir / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(paths)

    for png in plot.plot_sweep(sweep_dir):
        assert png.stat().st_size > 0
