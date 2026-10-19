"""Run the full `sweep` experiment and check its artifacts look reasonable."""

import csv
import json
from pathlib import Path

import pytest

from reef_perf.experiments.sweep import plot
from reef_perf.experiments.sweep.sweep import load_config, run_sweep
from reef_perf.microbenchmarks import list_microbenchmarks
from reef_perf.sim import MODULE_ROOT

ALL_CONFIG = MODULE_ROOT / "python/reef_perf/experiments/sweep/sweeps/all.yaml"


@pytest.mark.experiment
@pytest.mark.slow
def test_all_sweep_collects_every_artifact(
    cpp_build_dir: Path, tmp_path: Path
) -> None:
    """sweeps/all.yaml gives a sane artifact per microbenchmark, plus plots."""
    del cpp_build_dir  # built for its side effect
    config = load_config(ALL_CONFIG)
    assert config.microbenchmarks == list_microbenchmarks()

    paths = run_sweep(config, results_root=tmp_path)
    sweep_dir = tmp_path / config.name
    assert len(paths) == len(config.microbenchmarks) * len(config.variants)

    for path in paths:
        artifact = json.loads(path.read_text())
        result = artifact["result"]
        name = artifact["microbenchmark"]
        assert result["instructions"] > 0, name
        assert result["instructions"] == result["functional_instructions"], name
        assert result["cycles"] >= result["instructions"] / 4, name
        assert 0.0 < result["ipc"] <= 4.0, name

    with (sweep_dir / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(paths)

    for png in plot.plot_sweep(sweep_dir):
        assert png.stat().st_size > 0
