"""Check the correlation experiment and the report the PR bot posts."""

import json
from pathlib import Path

import pytest

from reef_perf.experiments.correlation.main import (
    GOLDEN,
    REPORT_MARKER,
    Row,
    correlate,
    load_rtl_cycles,
    main,
    render_markdown,
)
from reef_perf.microbenchmarks import list_microbenchmarks


@pytest.mark.unit
def test_rtl_cycles_lists_every_microbenchmark() -> None:
    """rtl_cycles.yaml has one entry per microbenchmark."""
    assert sorted(load_rtl_cycles()) == list_microbenchmarks()


@pytest.mark.unit
def test_report_shows_error_and_delta() -> None:
    """The report carries the marker, the ROI error and the cycle change."""
    rows = [
        Row("measured", 110, 100, 1000, 900, 1.5),
        Row("unmeasured", 50, None, 500, 500, 2.0),
    ]
    report = render_markdown(rows)
    assert report.startswith(REPORT_MARKER)
    assert (
        "| `measured` | 110 | 100 | +10.0% | 1000 | +100 (+11.1%) |" in report
    )
    assert "| `unmeasured` | 50 | - | - | 500 | 0 |" in report
    assert "Mean absolute ROI error:** 10.0% over 1 of 2" in report
    assert "Timing changed** on 1 of 2" in report


@pytest.mark.experiment
@pytest.mark.slow
def test_correlation_matches_golden(
    cpp_build_dir: Path, tmp_path: Path
) -> None:
    """Every microbenchmark has an ROI, and its cycles match the golden file."""
    del cpp_build_dir  # built for its side effect
    rows = correlate(baseline=json.loads(GOLDEN.read_text()))
    assert [row.microbenchmark for row in rows] == list_microbenchmarks()
    for row in rows:
        assert row.model_roi_cycles is not None, row.microbenchmark
        assert 0 < row.model_roi_cycles <= row.model_cycles, row.microbenchmark
        assert row.model_cycles == row.baseline_cycles, row.microbenchmark

    assert main(["--out", str(tmp_path)]) == 0
    assert (tmp_path / "report.md").read_text().startswith(REPORT_MARKER)
    assert len(json.loads((tmp_path / "correlation.json").read_text())) == len(
        rows
    )
