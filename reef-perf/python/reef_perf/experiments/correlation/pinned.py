"""Score the model against the pinned CoralNPU RTL (8a5e1b31).

Example:
    uv run python -m reef_perf.experiments.correlation.pinned

Runs the core, frontend and vector-unit microbenchmarks and the kernel ELFs
on m3.yaml + configs/machines/coral_pinned.yaml and compares each
region-of-interest cycle count with the pinned RTL's (rtl_pinned.csv next to
each rtl_m3.csv). The report also shows the M3 RTL's count, so a row says
both how far the RTL moved since M3 and how much of that the model follows.
Writes pinned.json and report.md under results/correlation_pinned/.

--no-overlay scores plain m3.yaml against the same RTL: the model with every
switch of coral_pinned.yaml off.
"""

import argparse
import csv
import json
import os
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from reef_perf.microbenchmarks import MICROBENCHMARK_DIR, build_microbenchmark
from reef_perf.sim import MODULE_ROOT, find_binary, run_sim

PINNED_CONFIG = MODULE_ROOT / "configs" / "machines" / "coral_pinned.yaml"
"""The pinned RTL's differences from M3, applied after m3.yaml."""

PINNED_BINARY = "reef_perf_pinned"
"""The simulator built with Spike as CoralNPU 8a5e1b31 builds it
(third_party/spike-pinned/README.md)."""


def find_pinned_binary() -> Path:
    """Locate reef_perf_pinned, the simulator for the pinned config.

    Falls back to reef_perf, with a warning, where it is not built (no
    pinned Spike installed). The two give identical results on every
    program today; the fallback keeps the pinned tests running there.

    Returns:
        Path to the executable.
    """
    try:
        return find_binary(PINNED_BINARY)
    except FileNotFoundError:
        warnings.warn(
            f"{PINNED_BINARY} is not built; running the pinned config on "
            "reef_perf (M3's Spike). Build the pinned Spike with "
            "scripts/build_deps.sh build/deps spike-pinned.",
            stacklevel=2,
        )
        return find_binary()


KERNEL_DIR = MODULE_ROOT / "kernel_correlation"
"""The kernel ELFs and their RTL measurements."""

SUITES = {
    "core": MICROBENCHMARK_DIR,
    "frontend": MICROBENCHMARK_DIR / "frontend",
    "vector_units": MICROBENCHMARK_DIR / "vector_units",
}
"""Microbenchmark suites: a directory of sources with its RTL CSVs."""

OUT_OF_SCOPE = {
    ("core", "fp_chain"),
    ("core", "fp_mix"),
    ("kernels", "attention"),
    ("kernels", "layernorm"),
}
"""Floating-point programs, not scored for now: Reef will most likely have no
scalar FPU (docs/pinned.md, "Out of scope"). They still run and are listed."""

RESULTS_ROOT = MODULE_ROOT / "results" / "correlation_pinned"
"""Where the experiment writes artifacts by default."""


@dataclass(frozen=True)
class Row:
    """One program on the model and on both RTLs.

    Attributes:
        suite: "core", "frontend", "vector_units" or "kernels".
        name: Program name.
        model_roi_cycles: Model cycles between the mcycle reads, if any.
        model_roi_instructions: Model instructions between the reads.
        rtl_roi_cycles: Pinned RTL cycles between the same reads.
        rtl_roi_instructions: Pinned RTL instructions between the reads.
        m3_rtl_roi_cycles: M3 RTL cycles, if measured.
    """

    suite: str
    name: str
    model_roi_cycles: int | None
    model_roi_instructions: int | None
    rtl_roi_cycles: int
    rtl_roi_instructions: int
    m3_rtl_roi_cycles: int | None

    @property
    def same_region(self) -> bool:
        """Whether the model and the RTL retired the same ROI instructions."""
        return self.model_roi_instructions == self.rtl_roi_instructions

    @property
    def in_scope(self) -> bool:
        """Whether the program counts toward the score (see OUT_OF_SCOPE)."""
        return (self.suite, self.name) not in OUT_OF_SCOPE

    @property
    def error(self) -> float | None:
        """Relative ROI error (model - RTL) / RTL, if the regions match."""
        if self.model_roi_cycles is None or not self.same_region:
            return None
        return (self.model_roi_cycles - self.rtl_roi_cycles) / (
            self.rtl_roi_cycles
        )


def read_rtl(path: Path) -> dict[str, tuple[int, int]]:
    """Read one RTL measurement file.

    Args:
        path: CSV whose first column is the program name, with
            rtl_roi_cycles and rtl_retired_insts columns.

    Returns:
        ROI cycles and instructions by program; unmeasured rows are left out.
    """
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        key = (reader.fieldnames or [""])[0]
        return {
            row[key]: (
                int(row["rtl_roi_cycles"]),
                int(row["rtl_retired_insts"]),
            )
            for row in reader
            if row["rtl_roi_cycles"]
        }


def list_programs() -> list[tuple[str, str]]:
    """List every program the pinned RTL is compared on.

    Returns:
        (suite, name) pairs: each suite's sources, then the kernel ELFs.
    """
    programs = [
        (suite, path.stem)
        for suite, directory in SUITES.items()
        for path in sorted(directory.glob("*.S"))
    ]
    programs += [
        ("kernels", path.stem)
        for path in sorted((KERNEL_DIR / "elfs").glob("*.elf"))
    ]
    return programs


def build_programs(out_dir: Path) -> dict[tuple[str, str], Path]:
    """Assemble the microbenchmarks and locate the kernel ELFs.

    Args:
        out_dir: Directory for the assembled microbenchmarks.

    Returns:
        ELF path by (suite, name).
    """
    elfs = {}
    for suite, name in list_programs():
        if suite == "kernels":
            elfs[suite, name] = KERNEL_DIR / "elfs" / f"{name}.elf"
            continue
        subdir = SUITES[suite].relative_to(MICROBENCHMARK_DIR)
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)
        elfs[suite, name] = build_microbenchmark(
            (subdir / name).as_posix(), out_dir
        )
    return elfs


def correlate(
    elf_dir: Path,
    overlay_configs: Sequence[Path] = (PINNED_CONFIG,),
    jobs: int | None = None,
    binary: Path | None = None,
) -> list[Row]:
    """Run every program and compare it with the pinned RTL.

    Args:
        elf_dir: Directory for the assembled microbenchmarks.
        overlay_configs: Configs applied after m3.yaml.
        jobs: Parallel simulator processes; one per CPU by default.
        binary: Simulator; reef_perf_pinned by default.

    Returns:
        One row per program with a pinned RTL measurement.
    """
    pinned = {}
    m3 = {}
    for suite, directory in (*SUITES.items(), ("kernels", KERNEL_DIR)):
        for name, measured in read_rtl(directory / "rtl_pinned.csv").items():
            pinned[suite, name] = measured
        for name, measured in read_rtl(directory / "rtl_m3.csv").items():
            m3[suite, name] = measured
    elfs = build_programs(elf_dir)
    keys = [key for key in elfs if key in pinned]
    with ThreadPoolExecutor(max_workers=jobs or os.cpu_count()) as pool:
        results = list(
            pool.map(
                lambda key: run_sim(
                    elfs[key],
                    overlay_configs=overlay_configs,
                    binary=binary or find_pinned_binary(),
                ),
                keys,
            )
        )
    return [
        Row(
            suite=key[0],
            name=key[1],
            model_roi_cycles=result.roi_cycles,
            model_roi_instructions=result.roi_instructions,
            rtl_roi_cycles=pinned[key][0],
            rtl_roi_instructions=pinned[key][1],
            m3_rtl_roi_cycles=m3[key][0] if key in m3 else None,
        )
        for key, result in zip(keys, results, strict=True)
    ]


def mean_abs_error(rows: Sequence[Row]) -> float | None:
    """Average the absolute ROI error over the rows that have one.

    Args:
        rows: Any rows.

    Returns:
        The mean, or None if no row has an error.
    """
    errors = [abs(row.error) for row in rows if row.error is not None]
    return sum(errors) / len(errors) if errors else None


def render_markdown(rows: Sequence[Row], title: str) -> str:
    """Render the comparison as a Markdown report.

    Args:
        rows: Output of correlate().
        title: Report heading.

    Returns:
        A summary table by suite, then one table per suite.
    """
    suites = list(dict.fromkeys(row.suite for row in rows))
    lines = [
        f"## {title}",
        "",
        "ROI cycles on the model and on the CoralNPU `8a5e1b31` RTL "
        "(`rtl_pinned.csv`). Error = (model - RTL) / RTL. RTL vs M3 = how far "
        "the RTL itself moved from M3 (`rtl_m3.csv`). Floating-point "
        "programs are listed but not scored (OUT_OF_SCOPE).",
        "",
        "| Suite | Programs | Mean abs error | Within 3% | Worst |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for suite in [*suites, "all"]:
        scored = [
            row
            for row in rows
            if row.error is not None
            and row.in_scope
            and suite in (row.suite, "all")
        ]
        mean = mean_abs_error(scored)
        if mean is None:
            continue
        worst = max(scored, key=lambda row: abs(row.error or 0.0))
        within = sum(abs(row.error or 0.0) <= 0.03 for row in scored)
        lines.append(
            f"| {suite} | {len(scored)} | {mean:.1%} | {within} "
            f"| `{worst.name}` {worst.error:+.1%} |"
        )
    for suite in suites:
        lines += [
            "",
            f"### {suite}",
            "",
            "| Program | Model ROI | RTL ROI | Error | RTL vs M3 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            if row.suite != suite:
                continue
            error = "insts differ" if row.error is None else f"{row.error:+.1%}"
            if not row.in_scope:
                error += " (not scored)"
            moved = "-"
            if row.m3_rtl_roi_cycles:
                moved = f"{row.rtl_roi_cycles / row.m3_rtl_roi_cycles - 1:+.1%}"
            lines.append(
                f"| `{row.name}` | {row.model_roi_cycles} "
                f"| {row.rtl_roi_cycles} | {error} | {moved} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Correlate, write the artifacts and print the report.

    Args:
        argv: Arguments, or None for sys.argv.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="score m3.yaml alone (every coral_pinned.yaml switch off)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_ROOT,
        help="output directory (default: results/correlation_pinned)",
    )
    parser.add_argument(
        "--jobs", type=int, default=None, help="parallel runs (default: CPUs)"
    )
    args = parser.parse_args(argv)
    overlays = () if args.no_overlay else (PINNED_CONFIG,)
    config = "m3.yaml" if args.no_overlay else "m3.yaml + coral_pinned.yaml"
    args.out.mkdir(parents=True, exist_ok=True)
    # m3.yaml alone is M3's machine, so it runs on M3's Spike.
    binary = find_binary() if args.no_overlay else find_pinned_binary()
    rows = correlate(args.out / "elfs", overlays, args.jobs, binary)
    report = render_markdown(rows, f"Model ({config}) against the pinned RTL")
    (args.out / "pinned.json").write_text(
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
