"""Check the kernel correlation report without running the simulator."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from reef_perf.experiments.correlation import kernels

RTL_CSV = """kernel,rtl_roi_cycles,rtl_retired_insts,notes
fast,1000,500,
slow,2000,800,
skewed,3000,900,
"""

MODEL = {
    "fast": (1100, 500),  # +10.0%
    "slow": (1900, 800),  # -5.0%
    "skewed": (3000, 901),  # Instruction counts differ.
}


@pytest.fixture
def kernel_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A kernel directory whose simulator results come from MODEL."""
    (tmp_path / "rtl_m3.csv").write_text(RTL_CSV)

    def fake_run_sim(elf: Path, config: Path) -> SimpleNamespace:
        cycles, insts = MODEL[elf.stem]
        return SimpleNamespace(roi_cycles=cycles, roi_instructions=insts)

    monkeypatch.setattr(kernels, "run_sim", fake_run_sim)
    return tmp_path


@pytest.mark.unit
def test_report_rows_and_mean(kernel_dir: Path) -> None:
    """Matching kernels get an error; the mean covers only those."""
    report = kernels.make_report(kernel_dir)
    assert "| `fast` | 1100 | 1000 | +10.0% | 500 |" in report
    assert "| `slow` | 1900 | 2000 | -5.0% | 800 |" in report
    assert "| `skewed` | 3000 | 3000 | insts differ | 901 |" in report
    assert "**Mean absolute ROI error:** 7.5% over 2 kernels." in report


@pytest.mark.unit
def test_no_matching_kernel_means_no_mean(
    kernel_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With every kernel mismatched, the report has no mean line."""
    monkeypatch.setitem(MODEL, "fast", (1100, 1))
    monkeypatch.setitem(MODEL, "slow", (1900, 1))
    assert "Mean absolute ROI error" not in kernels.make_report(kernel_dir)


@pytest.mark.unit
def test_append_keeps_the_existing_report(
    kernel_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--append adds the section after the microbenchmark report."""
    report = tmp_path / "report.md"
    report.write_text("## Microbenchmarks\n")
    monkeypatch.setattr(kernels, "make_report", lambda: "### Kernels\n")
    monkeypatch.setattr(sys, "argv", ["kernels", "--append", str(report)])
    kernels.main()
    assert report.read_text() == "## Microbenchmarks\n\n### Kernels\n"
