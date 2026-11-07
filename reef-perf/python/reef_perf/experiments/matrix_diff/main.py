r"""Per-iteration matrix costs on the model and the VME RTL, by differencing.

Example (the model steps run in the Docker image, the RTL step on the host,
where the CoralNPU simulator was built):

    uv run python -m reef_perf.experiments.matrix_diff.main build
    uv run python -m reef_perf.experiments.matrix_diff.main model
    PYTHONPATH=python python3 -m reef_perf.experiments.matrix_diff.main \
        rtl --sim <coralnpu>/bazel-bin/tests/verilator_sim/<vme_sim>
    uv run python -m reef_perf.experiments.matrix_diff.main compare

Each looping matrix microbenchmark is built twice, with N and 2N iterations
(the ITERS symbol). (ROI(2N) - ROI(N)) / N is the cost of one iteration with
the one-time costs (setup, the tail still in flight at ROI_END, and most of
the M3-vs-CoralNPU core difference) cancelled, which is what the matrix
parameters are calibrated against (docs/matrix.md, "Correlating it").

<vme_sim> is vme_core_mini_verification_axi_sim. Writes under
results/matrix_diff/: elfs/<n>/<name>.elf, model.csv, rtl.csv
and compare.csv. `rtl --record` also copies rtl.csv to
microbenchmarks/matrix/rtl_vme.csv, the committed reference.
"""

import argparse
import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from reef_perf.experiments.correlation import rtl_trace
from reef_perf.sim import DEFAULT_SIM_CONFIG, MODULE_ROOT

RESULTS_ROOT = MODULE_ROOT / "results" / "matrix_diff"
"""Where the experiment writes artifacts by default."""

VME_CONFIG = MODULE_ROOT / "configs" / "machines" / "m3_vme.yaml"
"""Turns Zvt on (applied after m3.yaml)."""

RTL_REFERENCE = MODULE_ROOT / "microbenchmarks" / "matrix" / "rtl_vme.csv"
"""Committed VME RTL measurements, one row per (program, iterations)."""

REFERENCE_CORE_PARAMS = {"top.core.rob.params.rob_entries": 16}
"""Core settings of the reference RTL that differ from M3.

The VME core is CoralNPU 8a5e1b31, whose retirement buffer has 16 entries
(Parameters.scala:141); M3's has 8. A Zvt instruction holds its entry until
its tile write lands, so the ROB size bounds how many MACs are in flight.
These are core parameters, owned by the core module: they are set here to
match the reference, not changed in m3.yaml or m3_vme.yaml.
"""


@dataclass(frozen=True)
class DiffBenchmark:
    """A matrix microbenchmark measured by differencing.

    Attributes:
        name: Program in microbenchmarks/matrix/.
        iters: N, the default ITERS; the second build uses 2N.
        ops_per_iter: Matrix operations of interest per loop iteration.
        op: What one operation is, for the report.
    """

    name: str
    iters: int
    ops_per_iter: int
    op: str


BENCHMARKS = (
    DiffBenchmark("mac_chain", 200, 1, "dependent vtmmu"),
    DiffBenchmark("mac_indep", 50, 4, "independent vtmmu"),
    DiffBenchmark("tile_load", 20, 16, "vtle8"),
    DiffBenchmark("tile_store", 20, 16, "vtse8"),
    DiffBenchmark("tile_load_behind_mac", 50, 1, "vtmmu + 8 vtle8 + lw"),
    DiffBenchmark("tile_load_burst", 20, 16, "vtle8, unrolled"),
    DiffBenchmark("tile_store_burst", 20, 16, "vtse8, unrolled"),
    DiffBenchmark("tile_load32_burst", 20, 16, "vtle32 (4 beats)"),
    DiffBenchmark("tile_store32_burst", 20, 16, "vtse32 (4 beats)"),
    DiffBenchmark("vec_load_burst", 20, 16, "vle8 m1 (1 reg)"),
    DiffBenchmark("vec_store_burst", 20, 16, "vse8 m1 (1 reg)"),
    DiffBenchmark("vec_load4_burst", 20, 16, "vle8 m4 (4 regs)"),
    DiffBenchmark("vec_store4_burst", 20, 16, "vse8 m4 (4 regs)"),
)
"""The looping matrix microbenchmarks."""


def _elf(out: Path, bench: DiffBenchmark, iters: int) -> Path:
    return out / "elfs" / str(iters) / f"{bench.name}.elf"


def _variants(bench: DiffBenchmark) -> tuple[int, int]:
    return bench.iters, 2 * bench.iters


def build(out: Path) -> None:
    """Build every benchmark at N and 2N iterations."""
    from reef_perf.microbenchmarks import build_microbenchmark

    for bench in BENCHMARKS:
        for iters in _variants(bench):
            dest = _elf(out, bench, iters).parent
            (dest / "matrix").mkdir(parents=True, exist_ok=True)
            built = build_microbenchmark(
                f"matrix/{bench.name}", dest, {"ITERS": iters}
            )
            shutil.move(built, _elf(out, bench, iters))


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def model(out: Path) -> None:
    """Run every built ELF on the model.

    The machine is m3.yaml + m3_vme.yaml with REFERENCE_CORE_PARAMS.
    """
    from reef_perf.sim import run_sim

    rows: list[dict[str, object]] = []
    for bench in BENCHMARKS:
        for iters in _variants(bench):
            result = run_sim(
                _elf(out, bench, iters),
                sim_config=DEFAULT_SIM_CONFIG,
                params=REFERENCE_CORE_PARAMS,
                overlay_configs=(VME_CONFIG,),
            )
            rows.append(
                {
                    "name": bench.name,
                    "iters": iters,
                    "roi_cycles": result.roi_cycles,
                    "roi_instructions": result.roi_instructions,
                }
            )
    _write(out / "model.csv", rows)


def rtl(out: Path, sim: Path, record: bool) -> None:
    """Run every built ELF on the VME verification simulator."""
    rows: list[dict[str, object]] = []
    for bench in BENCHMARKS:
        for iters in _variants(bench):
            elf = _elf(out, bench, iters)
            log = elf.with_suffix(".rtl.log")
            with log.open("w") as f:
                subprocess.run(
                    [
                        str(sim),
                        "--binary",
                        str(elf),
                        "--backdoor_load",
                        "--instr_trace",
                    ],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=3600,
                )
            m = rtl_trace.measure_log(log)
            rows.append(
                {
                    "name": bench.name,
                    "iters": iters,
                    "roi_cycles": m.cycles,
                    "roi_instructions": m.retired_insts,
                }
            )
    _write(out / "rtl.csv", rows)
    if record:
        shutil.copy(out / "rtl.csv", RTL_REFERENCE)
        print(f"recorded {RTL_REFERENCE}")


def _read(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open() as f:
        return {(r["name"], int(r["iters"])): r for r in csv.DictReader(f)}


def compare(out: Path) -> None:
    """Print and write per-operation costs, model against RTL."""
    model_rows = _read(out / "model.csv")
    rtl_path = out / "rtl.csv"
    rtl_rows = _read(rtl_path if rtl_path.is_file() else RTL_REFERENCE)
    rows: list[dict[str, object]] = []
    for bench in BENCHMARKS:
        n, n2 = _variants(bench)
        ops = n * bench.ops_per_iter
        costs = {}
        for side, table in (("model", model_rows), ("rtl", rtl_rows)):
            diff = int(table[(bench.name, n2)]["roi_cycles"]) - int(
                table[(bench.name, n)]["roi_cycles"]
            )
            costs[side] = diff / ops
        rows.append(
            {
                "name": bench.name,
                "op": bench.op,
                "model_cycles_per_op": round(costs["model"], 3),
                "rtl_cycles_per_op": round(costs["rtl"], 3),
                "error_pct": round(
                    100 * (costs["model"] - costs["rtl"]) / costs["rtl"], 1
                ),
            }
        )
    for row in rows:
        print(
            f"{row['name']:<11} {row['op']:<18} model "
            f"{row['model_cycles_per_op']:>7}  "
            f"rtl {row['rtl_cycles_per_op']:>7}  "
            f"error {row['error_pct']:>6}%"
        )
    _write(out / "compare.csv", rows)


def main() -> None:
    """Parse the command line and run one step."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["build", "model", "rtl", "compare"])
    parser.add_argument("--out", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--sim", type=Path, help="VME simulator (rtl step)")
    parser.add_argument(
        "--record",
        action="store_true",
        help="rtl step: also update microbenchmarks/matrix/rtl_vme.csv",
    )
    args = parser.parse_args()
    if args.step == "build":
        build(args.out)
    elif args.step == "model":
        model(args.out)
    elif args.step == "rtl":
        if args.sim is None:
            parser.error("the rtl step needs --sim")
        rtl(args.out, args.sim.expanduser(), args.record)
    else:
        compare(args.out)


if __name__ == "__main__":
    main()
