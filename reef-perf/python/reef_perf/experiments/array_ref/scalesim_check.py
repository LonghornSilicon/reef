r"""Cross-check the array reference against SCALE-Sim on single-fold GEMMs.

SCALE-Sim is not a reef-perf dependency. Run this in a throwaway
environment, for example inside the Docker image:

    uv venv /tmp/ss --python 3.12
    VIRTUAL_ENV=/tmp/ss uv pip install scalesim==3.0.0
    PYTHONPATH=python /tmp/ss/bin/python \\
        -m reef_perf.experiments.array_ref.scalesim_check

Only SCALE-Sim's compute stage is used: its demand matrices have one row
per cycle, so their length is the fold's cycle count. (Its memory model
fails on this numpy, and memory is not what is being checked.)

SCALE-Sim skews and loads over the whole array, and runs folds one after
another without overlap, so the comparison uses shapes that fill the array
in one fold. There the reference, with no pipeline depth and single
buffering, must give the same cycles.
"""

import contextlib
import io
import tempfile
from pathlib import Path

from scalesim.compute import systolic_compute_os, systolic_compute_ws
from scalesim.scale_sim import scalesim

from reef_perf.experiments.array_ref.ref import Array, measure

CONFIG = """[general]
run_name = ref
[architecture_presets]
ArrayHeight: {rows}
ArrayWidth: {cols}
IfmapSramSzkB: 1024
FilterSramSzkB: 1024
OfmapSramSzkB: 1024
IfmapOffset: 0
FilterOffset: 10000000
OfmapOffset: 20000000
Dataflow: {df}
Bandwidth: 10
ReadRequestBuffer: 32
WriteRequestBuffer: 32
[layout]
IfmapCustomLayout: False
IfmapSRAMBankBandwidth: 10
IfmapSRAMBankNum: 10
IfmapSRAMBankPort: 2
FilterCustomLayout: False
FilterSRAMBankBandwidth: 10
FilterSRAMBankNum: 10
FilterSRAMBankPort: 2
[sparsity]
SparsitySupport: false
SparseRep: ellpack_block
OptimizedMapping: false
BlockSize: 8
RandomNumberGeneratorSeed: 40
[run_presets]
InterfaceBandwidth: USER
UseRamulatorTrace: False
"""

CASES = (
    # (dataflow, rows, cols, M, N, K): each fills the array in one fold.
    ("os", 16, 16, 16, 16, 1),
    ("os", 16, 16, 16, 16, 4),
    ("os", 16, 16, 16, 16, 16),
    ("os", 8, 8, 8, 8, 4),
    ("os", 8, 16, 8, 16, 4),
    ("ws", 16, 16, 1, 16, 16),
    ("ws", 16, 16, 4, 16, 16),
    ("ws", 16, 16, 16, 16, 16),
    ("ws", 16, 16, 32, 16, 16),
    ("ws", 4, 8, 16, 8, 4),
    ("ws", 8, 8, 8, 8, 8),
)


class _Stop(Exception):
    pass


_CYCLES: list[int] = []


def _capture(cls: type) -> None:
    original = cls.get_demand_matrices

    def patched(self: object) -> None:
        _, _, ofmap = original(self)
        _CYCLES.append(ofmap.shape[0])
        raise _Stop

    cls.get_demand_matrices = patched


def scalesim_cycles(
    df: str, rows: int, cols: int, m: int, n: int, k: int
) -> int:
    """SCALE-Sim's compute cycles for one M x N x K GEMM."""
    d = Path(tempfile.mkdtemp())
    (d / "c.cfg").write_text(CONFIG.format(rows=rows, cols=cols, df=df))
    (d / "t.csv").write_text(f"Layer, M, N, K,\nl0, {m}, {n}, {k},\n")
    (d / "l.csv").write_text("Layer name,\n")
    sim = scalesim(
        save_disk_space=True,
        verbose=False,
        config=str(d / "c.cfg"),
        topology=str(d / "t.csv"),
        layout=str(d / "l.csv"),
        input_type_gemm=True,
    )
    _CYCLES.clear()
    with contextlib.suppress(_Stop), contextlib.redirect_stdout(io.StringIO()):
        sim.run_scale(top_path=str(d))
    return _CYCLES[0]


def main() -> None:
    """Print SCALE-Sim's and the reference's cycles for each case."""
    _capture(systolic_compute_os.systolic_compute_os)
    _capture(systolic_compute_ws.systolic_compute_ws)
    mismatches = 0
    for df, rows, cols, m, n, k in CASES:
        arr = Array(
            rows,
            cols,
            1,
            "output" if df == "os" else "weight",
            "forwarded",
            "forwarded",
            "in_place" if df == "os" else "chain",
            depth=0,
        )
        ref = int(measure(arr, m, n, k)["latency"])
        theirs = scalesim_cycles(df, rows, cols, m, n, k)
        ok = ref == theirs
        mismatches += not ok
        print(
            f"{df} {rows}x{cols} {m}x{n}x{k}: reference {ref}, "
            f"SCALE-Sim {theirs}{'' if ok else '  MISMATCH'}"
        )
    print(f"{len(CASES) - mismatches} of {len(CASES)} match")


if __name__ == "__main__":
    main()
