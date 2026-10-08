"""Run the reef_perf simulator and read its results.

Each call to run_sim() starts its own simulator process in its own temporary
directory, so calls are independent and safe to run in parallel.
"""

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[2]
"""The reef-perf directory (contains CMakeLists.txt and configs/)."""

DEFAULT_SIM_CONFIG = MODULE_ROOT / "configs" / "m3.yaml"
"""Simulation config describing the Reef M3 baseline."""

BIN_DIR_ENV = "REEF_PERF_BIN_DIR"
"""Environment variable naming the directory that holds the binaries."""


class SimulationError(RuntimeError):
    """The simulator failed or produced no results."""


def find_binary(name: str = "reef_perf") -> Path:
    """Locate a reef-perf executable.

    Looks in $REEF_PERF_BIN_DIR, then build/bin under the module root, then
    on PATH.

    Args:
        name: Executable name, e.g. "reef_perf" or "reef_trace".

    Returns:
        Path to the executable.

    Raises:
        FileNotFoundError: If it cannot be found.
    """
    candidates = []
    if os.environ.get(BIN_DIR_ENV):
        candidates.append(Path(os.environ[BIN_DIR_ENV]) / name)
    candidates.append(MODULE_ROOT / "build" / "bin" / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    found = shutil.which(name)
    if found:
        return Path(found)
    raise FileNotFoundError(
        f"{name} not found; build it with cmake (see README.md) or set "
        f"{BIN_DIR_ENV}"
    )


@dataclass(frozen=True)
class SimResult:
    """End-of-run summary of one simulation.

    Attributes:
        workload: Path of the ELF that was run.
        instructions: Instructions retired by the timing model.
        functional_instructions: Instructions executed by Spike. Equal to
            instructions unless the timing model deadlocked.
        cycles: Total cycles.
        ipc: Instructions per cycle.
        taken_redirects: Taken branches, jumps and traps.
        dispatch: Dispatch counters: cycles_with_dispatch and one
            stall_<reason> entry per stall reason.
        pools: Per execution resource: units, ops and busy_cycles.
    """

    workload: str
    instructions: int
    functional_instructions: int
    cycles: int
    ipc: float
    taken_redirects: int
    dispatch: dict[str, int]
    pools: dict[str, dict[str, int]]

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "SimResult":
        """Build a result from the simulator's --json output.

        Args:
            data: Parsed JSON object.

        Returns:
            The result.
        """
        return cls(
            workload=str(data["workload"]),
            instructions=int(data["instructions"]),
            functional_instructions=int(data["functional_instructions"]),
            cycles=int(data["cycles"]),
            ipc=float(data["ipc"]),
            taken_redirects=int(data["taken_redirects"]),
            dispatch={k: int(v) for k, v in data["dispatch"].items()},
            pools={
                name: {k: int(v) for k, v in counters.items()}
                for name, counters in data["pools"].items()
            },
        )

    def to_json(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable dictionary.

        Returns:
            The same layout as the simulator's --json output.
        """
        return asdict(self)

    @property
    def stall_cycles(self) -> dict[str, int]:
        """Cycles with no dispatch, keyed by stall reason."""
        prefix = "stall_"
        return {
            key[len(prefix) :]: value
            for key, value in self.dispatch.items()
            if key.startswith(prefix)
        }


def run_sim(
    elf: Path,
    sim_config: Path | None = DEFAULT_SIM_CONFIG,
    params: Mapping[str, object] | None = None,
    binary: Path | None = None,
    timeout: float | None = None,
) -> SimResult:
    """Run one workload on the simulator.

    Args:
        elf: RISC-V ELF to run.
        sim_config: Sparta configuration file, or None for built-in defaults.
        params: Parameter overrides, e.g.
            {"top.frontend.fetch.params.fetch_interval": 2}.
        binary: Simulator executable; found with find_binary() by default.
        timeout: Seconds before the run is abandoned.

    Returns:
        The simulator's summary.

    Raises:
        SimulationError: If the simulator fails.
    """
    exe = binary or find_binary()
    with tempfile.TemporaryDirectory(prefix="reef_perf_") as tmp:
        json_path = Path(tmp) / "result.json"
        cmd = [str(exe), "--elf", str(Path(elf).resolve())]
        cmd += ["--json", str(json_path)]
        if sim_config is not None:
            cmd += ["-c", str(Path(sim_config).resolve())]
        for key, value in (params or {}).items():
            cmd += ["-p", key, str(value)]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tmp,
            check=False,
        )
        if proc.returncode != 0 or not json_path.is_file():
            raise SimulationError(
                f"{' '.join(cmd)} failed with code {proc.returncode}:\n"
                f"{proc.stdout}\n{proc.stderr}"
            )
        return SimResult.from_json(json.loads(json_path.read_text()))
