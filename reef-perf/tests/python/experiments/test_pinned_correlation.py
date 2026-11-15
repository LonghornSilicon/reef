"""Check the pinned-RTL reference files and the report built from them."""

import pytest

from reef_perf.experiments.correlation.pinned import (
    KERNEL_DIR,
    PINNED_CONFIG,
    SUITES,
    Row,
    list_programs,
    read_rtl,
    render_markdown,
)

REFERENCES = {**SUITES, "kernels": KERNEL_DIR}


@pytest.mark.unit
def test_pinned_rtl_covers_every_program() -> None:
    """Each suite's rtl_pinned.csv has a row for each of its programs."""
    measured = {
        (suite, name)
        for suite, directory in REFERENCES.items()
        for name in read_rtl(directory / "rtl_pinned.csv")
    }
    assert measured == set(list_programs())


@pytest.mark.unit
@pytest.mark.parametrize("suite", sorted(REFERENCES))
def test_both_rtls_ran_the_same_instructions(suite: str) -> None:
    """The pinned and M3 RTLs retired the same ROI instructions.

    Both ran the same ELFs, so a difference means a file is stale.
    """
    directory = REFERENCES[suite]
    pinned = read_rtl(directory / "rtl_pinned.csv")
    m3 = read_rtl(directory / "rtl_m3.csv")
    assert {name: insts for name, (_, insts) in pinned.items()} == {
        name: insts for name, (_, insts) in m3.items()
    }


@pytest.mark.unit
def test_pinned_config_exists() -> None:
    """The overlay the golden test and the experiment apply is there."""
    assert PINNED_CONFIG.is_file()


@pytest.mark.unit
def test_report_scores_each_suite() -> None:
    """The report gives the error, the RTL's own move and a suite mean."""
    rows = [
        Row("core", "fast", 110, 50, 100, 50, 200),
        Row("core", "exact", 100, 50, 100, 50, 100),
        Row("kernels", "skewed", 90, 49, 100, 50, None),
    ]
    report = render_markdown(rows, "title")
    assert "| `fast` | 110 | 100 | +10.0% | -50.0% |" in report
    assert "| `skewed` | 90 | 100 | insts differ | - |" in report
    assert "| core | 2 | 5.0% | 1 | `fast` +10.0% |" in report
    assert "| all | 2 | 5.0% | 1 | `fast` +10.0% |" in report
    assert "| kernels |" not in report
