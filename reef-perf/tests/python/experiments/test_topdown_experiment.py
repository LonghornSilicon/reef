"""Run the top-down experiment and check its slot accounting."""

import json
from pathlib import Path

import pytest

from reef_perf.experiments.topdown.main import LEVEL2, analyze, main
from reef_perf.microbenchmarks import list_microbenchmarks
from reef_perf.sim import run_sim


@pytest.mark.experiment
@pytest.mark.slow
def test_every_slot_is_accounted_for(
    cpp_build_dir: Path, microbenchmark_elfs: dict[str, Path]
) -> None:
    """Retiring slots are the instructions; all slots fill whole cycles."""
    del cpp_build_dir  # built for its side effect
    names = list_microbenchmarks()
    trees = analyze(names)
    for name in names:
        result = run_sim(microbenchmark_elfs[name])
        width = result.topdown["dispatch_width"]
        tree = trees[name]
        assert tree.find("retiring").slots == result.instructions, name
        assert tree.slots % width == 0, name
        # Slots cover the cycles dispatch ran: all but the drain after the
        # last dispatch, at most the longest latency (div_chain: 33).
        assert result.cycles - 64 <= tree.slots // width <= result.cycles


@pytest.mark.experiment
@pytest.mark.slow
def test_writes_artifacts(cpp_build_dir: Path, tmp_path: Path) -> None:
    """main() writes a tree per microbenchmark, the table and the chart."""
    del cpp_build_dir  # built for its side effect
    args = ["--out", str(tmp_path)]
    assert main([*args, "--microbenchmark", "alu_chain"]) == 0
    tree = json.loads((tmp_path / "alu_chain.json").read_text())
    assert tree["name"] == "slots"
    summary = (tmp_path / "summary.md").read_text()
    assert all(category in summary for category in LEVEL2)
    assert (tmp_path / "topdown.png").stat().st_size > 0
