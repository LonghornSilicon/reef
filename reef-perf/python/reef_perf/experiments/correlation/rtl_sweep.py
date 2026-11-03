r"""Generate disjoint train/test RTL cycle datasets from microbenchmark sweeps.

From reef-perf/, after building the M3 verification Verilator executable:

    uv run python -m reef_perf.experiments.correlation.rtl_sweep \\
        --simulator /path/to/rvv_core_mini_verification_axi_sim

Runs parameterized microbenchmarks on the *RTL*, not the reef_perf model.
"""

import argparse
import csv
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from reef_perf.experiments.correlation.rtl_trace import measure_log
from reef_perf.microbenchmarks import (
    MICROBENCHMARK_DIR,
    build_microbenchmark,
    validate_constraints,
)
from reef_perf.sim import MODULE_ROOT

DEFAULT_SWEEPS = MICROBENCHMARK_DIR / "sweeps.json"
DEFAULT_OUTPUT = MODULE_ROOT / "results" / "rtl_sweep"
PARAMETER_RE = re.compile(r"^\.ifndef ([A-Z][A-Z0-9_]*)$", re.MULTILINE)
SPLITS = ("train", "test")


@dataclass(frozen=True)
class Run:
    """One uniquely identified benchmark and parameter configuration."""

    name: str
    split: str
    variant: int
    params: dict[str, int]

    @property
    def id(self) -> str:
        """Path-safe identifier used for log and ELF output files."""
        return f"{self.name.replace('/', '__')}_{self.variant:02d}"


def load_sweeps(path: Path = DEFAULT_SWEEPS) -> list[Run]:
    """Load a sweep JSON file, checking all benchmarks and split isolation.

    A missing benchmark, unexpected parameter, duplicate configuration, or
    common train/test configuration is an error rather than a silent gap.
    """
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("sweeps must be a JSON object")
    sources = {
        p.relative_to(MICROBENCHMARK_DIR).with_suffix("").as_posix(): p
        for p in MICROBENCHMARK_DIR.rglob("*.S")
    }
    if set(raw) != set(sources):
        missing = sorted(set(sources) - set(raw))
        unknown = sorted(set(raw) - set(sources))
        raise ValueError(
            f"sweep/source mismatch: missing={missing}; unknown={unknown}"
        )
    runs = []
    for name, source in sorted(sources.items()):
        allowed = set(PARAMETER_RE.findall(source.read_text()))
        if not allowed:
            raise ValueError(f"{name}: no declared parameters")
        splits = raw[name]
        if not isinstance(splits, dict) or set(splits) != set(SPLITS):
            raise ValueError(f"{name}: expected train and test lists")
        seen = set()
        for split in SPLITS:
            entries = splits[split]
            if not isinstance(entries, list) or not entries:
                raise ValueError(f"{name}/{split}: expected nonempty list")
            for variant, params in enumerate(entries):
                if not isinstance(params, dict) or set(params) != allowed:
                    raise ValueError(
                        f"{name}/{split}/{variant}: expected parameters "
                        f"{sorted(allowed)}"
                    )
                if any(
                    isinstance(v, bool) or not isinstance(v, int) or v <= 0
                    for v in params.values()
                ):
                    raise ValueError(
                        f"{name}/{split}/{variant}: values must be positive "
                        "integers"
                    )
                validate_constraints(name, params)
                key = tuple(sorted(params.items()))
                if key in seen:
                    raise ValueError(
                        f"{name}: duplicate train/test configuration: {params}"
                    )
                seen.add(key)
                runs.append(Run(name, split, variant, params))
    return runs


def run_rtl(
    simulator: Path, elf: Path, log: Path, timeout: int
) -> tuple[int, int]:
    """Run the M3 verification simulator and measure the traced ROI."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        subprocess.run(
            [
                str(simulator.resolve()),
                "--binary",
                str(elf.resolve()),
                "--backdoor_load",
                "--instr_trace",
            ],
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
        )
    measurement = measure_log(log)
    if not measurement.reached_mpause:
        raise ValueError(f"{log}: RTL did not retire mpause")
    if measurement.cycles <= 0 or measurement.retired_insts <= 0:
        raise ValueError(f"{log}: empty ROI measurement")
    return measurement.cycles, measurement.retired_insts


def generate(
    runs: list[Run],
    simulator: Path | None,
    output: Path,
    timeout: int = 600,
    compile_only: bool = False,
    csv_dir: Path | None = None,
) -> dict[str, Path]:
    """Build and optionally measure all cases; write CSVs only with RTL data."""
    if not compile_only and (simulator is None or not simulator.is_file()):
        raise FileNotFoundError(f"Verilator simulator not found: {simulator}")
    output.mkdir(parents=True, exist_ok=True)
    csv_dir = output if csv_dir is None else csv_dir
    if not compile_only:
        csv_dir.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for run in runs for key in run.params})
    fields = [
        "microbenchmark",
        "variant",
        *columns,
        "rtl_roi_cycles",
        "rtl_retired_insts",
        "notes",
    ]
    rows: dict[str, list[dict]] = {split: [] for split in SPLITS}
    for run in runs:
        work = output / "elfs" / run.split / run.id
        elf = build_microbenchmark(run.name, work, run.params)
        if compile_only:
            print(f"compiled {run.split}/{run.id}: {run.params}")
            continue
        log = output / "logs" / run.split / f"{run.id}.log"
        try:
            cycles, insts = run_rtl(simulator, elf, log, timeout)
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            ValueError,
        ) as exc:
            raise RuntimeError(
                f"RTL failed for {run.split}/{run.id}; log: {log}"
            ) from exc
        rows[run.split].append(
            {
                "microbenchmark": run.name,
                "variant": run.variant,
                **run.params,
                "rtl_roi_cycles": cycles,
                "rtl_retired_insts": insts,
                "notes": "",
            }
        )
        print(
            f"{run.split}/{run.id}: {cycles} cycles, {insts} insts",
            flush=True,
        )

    result = {}
    if not compile_only:
        # Stage every requested dataset on the destination filesystem. Neither
        # existing CSV is touched unless *all* builds, RTL runs, and CSV writes
        # have succeeded. TemporaryDirectory cleans up on failure.
        with tempfile.TemporaryDirectory(
            prefix=".rtl_sweep_", dir=csv_dir
        ) as staging:
            completed = {}
            for split in SPLITS:
                if not rows[split]:
                    continue
                temp = Path(staging) / f"rtl_m3_{split}.csv"
                with temp.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows[split])
                completed[split] = temp
            for split, temp in completed.items():
                target = csv_dir / f"rtl_m3_{split}.csv"
                temp.replace(target)
                result[split] = target
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for compiling and measuring train/test RTL sweeps."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--simulator",
        type=Path,
        help="built M3 verification Verilator executable",
    )
    parser.add_argument("--sweeps", type=Path, default=DEFAULT_SWEEPS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=MICROBENCHMARK_DIR,
        help="output directory for tracked RTL CSV datasets",
    )
    parser.add_argument("--split", choices=("both", *SPLITS), default="both")
    parser.add_argument(
        "--timeout", type=int, default=600, help="seconds per RTL run"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="validate and list configurations without compiling",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="build ELF variants but do not run RTL or write CSVs",
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    runs = load_sweeps(args.sweeps)
    if args.split != "both":
        runs = [run for run in runs if run.split == args.split]
    if args.list:
        for run in runs:
            print(f"{run.split}/{run.id}: {run.params}")
        return 0
    results = generate(
        runs,
        args.simulator,
        args.out,
        args.timeout,
        args.compile_only,
        args.csv_dir,
    )
    for split, path in results.items():
        count = sum(run.split == split for run in runs)
        print(f"{split} dataset: {path} ({count} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
