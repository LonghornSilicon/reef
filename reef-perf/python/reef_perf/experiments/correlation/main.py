"""Correlate the model with the M3 RTL on every microbenchmark.

Example:
    uv run python -m reef_perf.experiments.correlation.main

Runs every microbenchmark on the M3 config and compares the model's
region-of-interest cycles and instruction counts with the RTL's
(microbenchmarks/rtl_m3.csv, measured with rtl_trace.py). Total cycles are
compared with a baseline in the golden-file format
(tests/golden/m3_results.json by default; the PR bot passes the base branch's
copy), so a PR shows how it moved the model. Writes correlation.json and a
Markdown report (the PR comment) under results/correlation/.
"""

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from reef_perf.experiments.sweep.main import ELF_DIR
from reef_perf.microbenchmarks import (
    build_microbenchmarks,
    list_microbenchmarks,
)
from reef_perf.sim import DEFAULT_SIM_CONFIG, MODULE_ROOT, run_sim

RTL_RESULTS = MODULE_ROOT / "microbenchmarks" / "rtl_m3.csv"
"""Region-of-interest measurements on the M3 RTL, one row per
microbenchmark."""

GOLDEN = MODULE_ROOT / "tests" / "golden" / "m3_results.json"
"""Default baseline: the golden timing file."""

RESULTS_ROOT = MODULE_ROOT / "results" / "correlation"
"""Where the correlation experiment writes artifacts by default."""

REPORT_MARKER = "<!-- reef-perf-correlation -->"
"""First line of the report; the PR bot finds its own comment by it."""


@dataclass(frozen=True)
class RtlResult:
    """One microbenchmark's region of interest on the RTL.

    Attributes:
        roi_cycles: mcycle at ROI_END minus mcycle at ROI_BEGIN.
        roi_instructions: Instructions retired between the two reads.
    """

    roi_cycles: int
    roi_instructions: int


@dataclass(frozen=True)
class Row:
    """One microbenchmark's correlation.

    Attributes:
        microbenchmark: Microbenchmark name.
        model_roi_cycles: Model cycles between the mcycle reads, if any.
        rtl_roi_cycles: RTL cycles between the same reads, if measured.
        model_roi_instructions: Model instructions between the reads.
        rtl_roi_instructions: RTL instructions between the reads.
        model_cycles: Model cycles for the whole program.
        baseline_cycles: Baseline whole-program cycles, if in the baseline.
        ipc: Model instructions per cycle.
    """

    microbenchmark: str
    model_roi_cycles: int | None
    rtl_roi_cycles: int | None
    model_roi_instructions: int | None
    rtl_roi_instructions: int | None
    model_cycles: int
    baseline_cycles: int | None
    ipc: float

    @property
    def error(self) -> float | None:
        """Relative ROI error (model - RTL) / RTL, if both are known."""
        if self.model_roi_cycles is None or not self.rtl_roi_cycles:
            return None
        return (
            self.model_roi_cycles - self.rtl_roi_cycles
        ) / self.rtl_roi_cycles

    @property
    def same_region(self) -> bool:
        """Whether both sides retired the same instructions in the ROI.

        If not, the cycle counts measure different work and the error is
        meaningless. True when the RTL count is unknown.
        """
        return (
            self.rtl_roi_instructions is None
            or self.model_roi_instructions == self.rtl_roi_instructions
        )


def load_rtl_results(path: Path = RTL_RESULTS) -> dict[str, RtlResult]:
    """Read the RTL measurements.

    Args:
        path: CSV with the columns microbenchmark, rtl_roi_cycles,
            rtl_retired_insts and notes; a row with empty counts is not
            measured yet.

    Returns:
        The measured microbenchmarks.

    Raises:
        ValueError: If the file names an unknown microbenchmark.
    """
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    unknown = sorted(
        {row["microbenchmark"] for row in rows} - set(list_microbenchmarks())
    )
    if unknown:
        raise ValueError(f"unknown microbenchmarks in {path}: {unknown}")
    return {
        row["microbenchmark"]: RtlResult(
            int(row["rtl_roi_cycles"]), int(row["rtl_retired_insts"])
        )
        for row in rows
        if row["rtl_roi_cycles"]
    }


def correlate(
    sim_config: Path = DEFAULT_SIM_CONFIG,
    rtl: dict[str, RtlResult] | None = None,
    baseline: dict[str, dict] | None = None,
    jobs: int | None = None,
) -> list[Row]:
    """Run every microbenchmark and compare it with the RTL and a baseline.

    Args:
        sim_config: Simulation config.
        rtl: RTL measurements; read from RTL_RESULTS when omitted.
        baseline: Golden-format results to compare total cycles against.
        jobs: Parallel simulator processes; one per CPU by default.

    Returns:
        One row per microbenchmark, sorted by name.
    """
    rtl = load_rtl_results() if rtl is None else rtl
    baseline = baseline or {}
    names = list_microbenchmarks()
    elfs = build_microbenchmarks(names, ELF_DIR)
    with ThreadPoolExecutor(max_workers=jobs or os.cpu_count()) as pool:
        results = list(
            pool.map(lambda n: run_sim(elfs[n], sim_config=sim_config), names)
        )
    rows = []
    for name, result in zip(names, results, strict=True):
        measured = rtl.get(name)
        rows.append(
            Row(
                microbenchmark=name,
                model_roi_cycles=result.roi_cycles,
                rtl_roi_cycles=measured.roi_cycles if measured else None,
                model_roi_instructions=result.roi_instructions,
                rtl_roi_instructions=(
                    measured.roi_instructions if measured else None
                ),
                model_cycles=result.cycles,
                baseline_cycles=baseline.get(name, {}).get("cycles"),
                ipc=result.ipc,
            )
        )
    return rows


def format_delta(row: Row) -> str:
    """Describe how the model's total cycles moved against the baseline.

    Args:
        row: One microbenchmark's correlation.

    Returns:
        "new", "0", or a signed change with its percentage.
    """
    if row.baseline_cycles is None:
        return "new"
    delta = row.model_cycles - row.baseline_cycles
    if delta == 0:
        return "0"
    return f"{delta:+d} ({100 * delta / row.baseline_cycles:+.1f}%)"


def render_markdown(
    rows: list[Row], title: str = "reef-perf correlation"
) -> str:
    """Render the correlation as a Markdown report (the PR comment).

    Args:
        rows: Output of correlate().
        title: Report heading.

    Returns:
        The report, starting with REPORT_MARKER.
    """

    def show(value: int | None) -> str:
        return "-" if value is None else str(value)

    lines = [
        REPORT_MARKER,
        f"## {title}",
        "",
        "ROI = the loop between the microbenchmark's two `csrr mcycle` reads, "
        "on the model and on the M3 RTL (`microbenchmarks/rtl_m3.csv`). "
        "Error = (model - RTL) / RTL. Δ = change in total model cycles against "
        "the baseline.",
        "",
        "| Microbenchmark | Model ROI | RTL ROI | Error | ROI insts "
        "| Model cycles | Δ cycles | IPC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        error = "-" if row.error is None else f"{100 * row.error:+.1f}%"
        insts = show(row.model_roi_instructions)
        if not row.same_region:
            insts += f" (RTL {row.rtl_roi_instructions})"
        lines.append(
            f"| `{row.microbenchmark}` | {show(row.model_roi_cycles)} "
            f"| {show(row.rtl_roi_cycles)} | {error} | {insts} "
            f"| {row.model_cycles} | {format_delta(row)} | {row.ipc:.3f} |"
        )

    errors = [
        abs(row.error)
        for row in rows
        if row.error is not None and row.same_region
    ]
    mismatched = [row.microbenchmark for row in rows if not row.same_region]
    changed = sum(format_delta(row) not in ("0", "new") for row in rows)
    lines.append("")
    if errors:
        mean = 100 * sum(errors) / len(errors)
        lines.append(
            f"**Mean absolute ROI error:** {mean:.1f}% over {len(errors)} of "
            f"{len(rows)} microbenchmarks."
        )
    else:
        lines.append("**No RTL measurements** in microbenchmarks/rtl_m3.csv.")
    if mismatched:
        lines.append("")
        lines.append(
            "**ROI instruction counts differ from the RTL** on "
            + ", ".join(f"`{name}`" for name in mismatched)
            + ", so their cycle counts measure different work; they are left "
            "out of the mean."
        )
    lines.append("")
    lines.append(
        f"**Timing changed** on {changed} of {len(rows)} microbenchmarks."
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments, or None for sys.argv.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sim-config",
        type=Path,
        default=DEFAULT_SIM_CONFIG,
        help="simulation config (default: configs/machines/m3.yaml)",
    )
    parser.add_argument(
        "--rtl",
        type=Path,
        default=RTL_RESULTS,
        help="RTL measurements (default: microbenchmarks/rtl_m3.csv)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=GOLDEN,
        help="golden-format results (default: tests/golden/m3_results.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_ROOT,
        help="output directory (default: results/correlation)",
    )
    parser.add_argument("--title", default="reef-perf correlation")
    parser.add_argument(
        "--jobs", type=int, default=None, help="parallel runs (default: CPUs)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Correlate, write the artifacts and print the report.

    Args:
        argv: Arguments, or None for sys.argv.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    baseline = (
        json.loads(args.baseline.read_text()) if args.baseline.is_file() else {}
    )
    rows = correlate(
        args.sim_config, load_rtl_results(args.rtl), baseline, args.jobs
    )
    report = render_markdown(rows, args.title)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "correlation.json").write_text(
        json.dumps(
            [asdict(row) | {"error": row.error} for row in rows], indent=2
        )
        + "\n"
    )
    (args.out / "report.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
