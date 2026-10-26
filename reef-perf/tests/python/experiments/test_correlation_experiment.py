"""Check the correlation experiment and the report the PR bot posts."""

import json
from pathlib import Path

import pytest

from reef_perf.experiments.correlation.main import (
    GOLDEN,
    REPORT_MARKER,
    Row,
    correlate,
    load_rtl_results,
    main,
    render_markdown,
)
from reef_perf.microbenchmarks import list_microbenchmarks


@pytest.mark.unit
def test_rtl_results_cover_every_microbenchmark() -> None:
    """rtl_m3.csv has a measurement for every microbenchmark."""
    assert sorted(load_rtl_results()) == list_microbenchmarks()


@pytest.mark.unit
def test_report_shows_error_and_delta() -> None:
    """The report carries the marker, the ROI error and the cycle change."""
    rows = [
        Row("measured", 110, 100, 50, 50, 1000, 900, 1.5),
        Row("unmeasured", 50, None, 20, None, 500, 500, 2.0),
        Row("mismatched", 90, 100, 49, 50, 800, 800, 1.0),
    ]
    report = render_markdown(rows)
    assert report.startswith(REPORT_MARKER)
    assert (
        "| `measured` | 110 | 100 | +10.0% | 50 | 1000 | +100 (+11.1%) |"
        in report
    )
    assert "| `unmeasured` | 50 | - | - | 20 | 500 | 0 |" in report
    assert "| 49 (RTL 50) |" in report
    # The mismatched row is flagged and left out of the mean.
    assert "Mean absolute ROI error:** 10.0% over 1 of 3" in report
    assert "differ from the RTL** on `mismatched`" in report
    assert "Timing changed** on 1 of 3" in report


@pytest.mark.experiment
@pytest.mark.slow
def test_correlation_matches_golden_and_rtl_region(
    cpp_build_dir: Path, tmp_path: Path
) -> None:
    """The model's ROI is the RTL's ROI, and its cycles match the golden."""
    del cpp_build_dir  # built for its side effect
    rows = correlate(baseline=json.loads(GOLDEN.read_text()))
    assert [row.microbenchmark for row in rows] == list_microbenchmarks()
    for row in rows:
        name = row.microbenchmark
        assert row.model_roi_cycles is not None, name
        assert 0 < row.model_roi_cycles <= row.model_cycles, name
        assert row.model_cycles == row.baseline_cycles, name
        assert row.same_region, (
            name,
            row.model_roi_instructions,
            row.rtl_roi_instructions,
        )
        assert row.error is not None, name

    assert main(["--out", str(tmp_path)]) == 0
    assert (tmp_path / "report.md").read_text().startswith(REPORT_MARKER)
    assert len(json.loads((tmp_path / "correlation.json").read_text())) == len(
        rows
    )
