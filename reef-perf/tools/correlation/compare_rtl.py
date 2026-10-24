"""Compare model ROI cycles with RTL measurements.

Runs every microbenchmark in microbenchmarks/rtl_m3.csv and every variant in
tools/correlation/rtl_variants.csv on one or more builds of the model, and
prints each one's ROI cycles (per loop iteration for variants) and error
against the RTL.

Usage (from reef-perf/, inside Docker):

    uv run python tools/correlation/compare_rtl.py
    uv run python tools/correlation/compare_rtl.py \
        --model old=/tmp/reef_perf_base:/tmp/m3_base.yaml \
        --model new=build/bin/reef_perf --only va1_
"""

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from reef_perf.microbenchmarks import (
    ASSEMBLER,
    LINKER,
    MARCH,
    MICROBENCHMARK_DIR,
)
from reef_perf.sim import DEFAULT_SIM_CONFIG, SimulationError, run_sim

TOOL_DIR = Path(__file__).resolve().parent
VARIANT_DIR = TOOL_DIR / "variants"


@dataclass(frozen=True)
class Benchmark:
    """One program with an RTL measurement.

    Attributes:
        name: Microbenchmark or variant name.
        source: Its assembly source.
        rtl_roi_cycles: ROI cycles on the M3 RTL.
        iterations: Loop iterations in the ROI; 1 for microbenchmarks.
    """

    name: str
    source: Path
    rtl_roi_cycles: int
    iterations: int


@dataclass(frozen=True)
class Model:
    """One build of the model to compare.

    Attributes:
        label: Column name in the output.
        binary: The reef_perf executable.
        config: Its machine configuration.
    """

    label: str
    binary: Path
    config: Path


def load_benchmarks() -> list[Benchmark]:
    """Read the RTL measurements of the microbenchmarks and variants."""
    benchmarks = []
    with (MICROBENCHMARK_DIR / "rtl_m3.csv").open() as f:
        for row in csv.DictReader(f):
            name = row["microbenchmark"]
            benchmarks.append(
                Benchmark(
                    name,
                    MICROBENCHMARK_DIR / f"{name}.S",
                    int(row["rtl_roi_cycles"]),
                    1,
                )
            )
    with (TOOL_DIR / "rtl_variants.csv").open() as f:
        for row in csv.DictReader(f):
            name = row["variant"]
            benchmarks.append(
                Benchmark(
                    name,
                    VARIANT_DIR / row["set"] / f"{name}.S",
                    int(row["rtl_roi_cycles"]),
                    int(row["iterations"]),
                )
            )
    return benchmarks


def assemble(source: Path, out_dir: Path) -> Path:
    """Assemble and link a source against the microbenchmarks' includes.

    Args:
        source: Assembly source.
        out_dir: Directory for the object file and ELF.

    Returns:
        The ELF.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    obj = out_dir / f"{source.stem}.o"
    elf = out_dir / f"{source.stem}.elf"
    subprocess.run(
        [
            ASSEMBLER,
            f"-march={MARCH}",
            "-mabi=ilp32f",
            "-I",
            str(MICROBENCHMARK_DIR),
            str(source),
            "-o",
            str(obj),
        ],
        check=True,
    )
    subprocess.run(
        [
            LINKER,
            "-m",
            "elf32lriscv",
            "-T",
            str(MICROBENCHMARK_DIR / "link.ld"),
            str(obj),
            "-o",
            str(elf),
        ],
        check=True,
    )
    return elf


def parse_model(text: str) -> Model:
    """Parse a --model argument: LABEL=BINARY[:CONFIG]."""
    label, _, rest = text.partition("=")
    binary, _, config = rest.partition(":")
    if not label or not binary:
        raise argparse.ArgumentTypeError(
            f"expected LABEL=BINARY[:CONFIG], got {text!r}"
        )
    return Model(
        label,
        Path(binary).resolve(),
        Path(config).resolve() if config else DEFAULT_SIM_CONFIG,
    )


def measure(model: Model, elf: Path) -> int | None:
    """Return the ROI cycles of an ELF on a model, or None if it fails."""
    try:
        return run_sim(
            elf, sim_config=model.config, binary=model.binary
        ).roi_cycles
    except SimulationError:
        return None


def main() -> int:
    """Run the comparison and print the table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        type=parse_model,
        action="append",
        help="LABEL=BINARY[:CONFIG]; repeat to compare builds "
        "(default: model=build/bin/reef_perf)",
    )
    parser.add_argument(
        "--only", default="", help="only names containing this text"
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("results/correlation_tools"),
        help="where to put the ELFs",
    )
    args = parser.parse_args()
    models = args.model or [parse_model("model=build/bin/reef_perf")]

    header = f"{'benchmark':<16} {'RTL':>9}"
    for model in models:
        header += f" {model.label:>9} {'err':>8}"
    print(header)
    for bench in load_benchmarks():
        if args.only not in bench.name:
            continue
        elf = assemble(bench.source, args.build_dir)
        rtl = bench.rtl_roi_cycles / bench.iterations
        line = f"{bench.name:<16} {rtl:>9.2f}"
        for model in models:
            cycles = measure(model, elf)
            if cycles is None:
                line += f" {'fail':>9} {'':>8}"
                continue
            value = cycles / bench.iterations
            line += f" {value:>9.2f} {100 * (value - rtl) / rtl:>+7.1f}%"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
