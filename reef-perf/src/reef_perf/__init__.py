"""Python interface to the reef_perf performance model.

The simulator itself is C++ (src/csrc/). This package runs it, reads its
results, and holds the experiments built on top of it.
"""

from reef_perf.sim import (
    DEFAULT_SIM_CONFIG,
    MODULE_ROOT,
    SimResult,
    SimulationError,
    find_binary,
    run_sim,
)

__all__ = [
    "DEFAULT_SIM_CONFIG",
    "MODULE_ROOT",
    "SimResult",
    "SimulationError",
    "find_binary",
    "run_sim",
]
