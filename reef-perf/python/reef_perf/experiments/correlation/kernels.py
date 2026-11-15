"""Correlate the model with the M3 RTL on the inference engine's kernels.

Example:
    uv run python -m reef_perf.experiments.correlation.kernels

Runs every kernel ELF in kernel_correlation/elfs/ on the M3 config and
compares its region-of-interest cycles and instructions with the RTL's
(kernel_correlation/rtl_m3.csv, measured on the same ELFs by
kernel_correlation/run_rtl.sh). Prints a Markdown section; the PR bot appends
it to the microbenchmark report.
"""

import argparse
import csv
from pathlib import Path

from reef_perf.sim import DEFAULT_SIM_CONFIG, MODULE_ROOT, run_sim

KERNEL_DIR = MODULE_ROOT / "kernel_correlation"
"""The kernel ELFs and their RTL measurements."""


def read_rtl(path: Path) -> dict[str, tuple[int, int]]:
    """Read the RTL measurements.

    Args:
        path: CSV with kernel, rtl_roi_cycles and rtl_retired_insts columns.

    Returns:
        ROI cycles and instructions by kernel name.
    """
    with path.open(newline="") as file:
        return {
            row["kernel"]: (
                int(row["rtl_roi_cycles"]),
                int(row["rtl_retired_insts"]),
            )
            for row in csv.DictReader(file)
        }


def make_report(kernel_dir: Path = KERNEL_DIR) -> str:
    """Run every kernel on the model and compare it with the RTL.

    Args:
        kernel_dir: Directory with elfs/ and rtl_m3.csv.

    Returns:
        The Markdown section.
    """
    lines = [
        "### Kernels",
        "",
        "The inference engine's kernels on the M3 as built (8 KB ITCM, "
        "32 KB DTCM), against `kernel_correlation/rtl_m3.csv`.",
        "",
        "| Kernel | Model ROI | RTL ROI | Error | ROI insts |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    errors = []
    for name, (rtl_cycles, rtl_insts) in sorted(
        read_rtl(kernel_dir / "rtl_m3.csv").items()
    ):
        elf = kernel_dir / "elfs" / f"{name}.elf"
        result = run_sim(elf, DEFAULT_SIM_CONFIG)
        if result.roi_cycles is None or result.roi_instructions != rtl_insts:
            lines.append(
                f"| `{name}` | {result.roi_cycles} | {rtl_cycles} | "
                f"insts differ | {result.roi_instructions} |"
            )
            continue
        error = (result.roi_cycles - rtl_cycles) / rtl_cycles
        errors.append(abs(error))
        lines.append(
            f"| `{name}` | {result.roi_cycles} | {rtl_cycles} | "
            f"{error:+.1%} | {rtl_insts} |"
        )
    if errors:
        lines += [
            "",
            f"**Mean absolute ROI error:** {sum(errors) / len(errors):.1%} "
            f"over {len(errors)} kernels.",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Print the kernel section, or append it to --append."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--append", type=Path, help="report to append to")
    args = parser.parse_args()
    report = make_report()
    if args.append:
        with args.append.open("a") as file:
            file.write("\n" + report)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
