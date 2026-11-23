"""The pinned-RTL config matches the pinned RTL on every in-scope program.

Runs the core, frontend and vector-unit microbenchmarks and the kernel ELFs
on m3.yaml + configs/machines/coral_pinned.yaml (reef_perf_pinned), as
`python -m reef_perf.experiments.correlation.pinned` does, and holds each
region-of-interest cycle count to TOLERANCE of rtl_pinned.csv. Floating-point
programs are not scored (pinned.OUT_OF_SCOPE). The pinned golden file pins
the model's own numbers; this pins them to the RTL.
"""

from pathlib import Path

import pytest

from reef_perf.experiments.correlation import pinned

TOLERANCE = 0.03
"""Allowed relative ROI-cycle error ("good enough", set 2026-11-15)."""

KNOWN_GAP: dict[str, str] = {}
"""Programs ("suite/name") with a known error above TOLERANCE, and why."""


@pytest.mark.integration
@pytest.mark.slow
def test_every_program_is_within_tolerance(
    cpp_build_dir: Path, tmp_path: Path
) -> None:
    """Every in-scope program is within 3% of the pinned RTL."""
    del cpp_build_dir  # built for its side effect
    rows = pinned.correlate(tmp_path)
    assert len(rows) == len(pinned.list_programs())
    problems = []
    for row in rows:
        name = f"{row.suite}/{row.name}"
        if not row.same_region:
            problems.append(
                f"{name}: ROI instructions {row.model_roi_instructions}, "
                f"RTL {row.rtl_roi_instructions}"
            )
        elif row.in_scope and name not in KNOWN_GAP:
            assert row.error is not None
            if abs(row.error) > TOLERANCE:
                problems.append(
                    f"{name}: {row.error:+.1%} ({row.model_roi_cycles} "
                    f"against {row.rtl_roi_cycles})"
                )
    assert not problems, "\n".join(problems)


@pytest.mark.unit
def test_known_gaps_are_programs() -> None:
    """Every known gap names a program the experiment runs."""
    names = {f"{suite}/{name}" for suite, name in pinned.list_programs()}
    assert set(KNOWN_GAP) <= names
