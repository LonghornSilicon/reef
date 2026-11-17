"""Each switch of coral_pinned.yaml works on its own.

The intermediate machines between M3 and the pinned RTL are built by turning
some of these switches on and leaving the rest off. Each switch, with only
the switches it needs (REQUIRES), runs on m3.yaml + m3_vme.yaml with no
deadlock or error and changes the timing of at least one program, unless
NO_EFFECT says why it does not.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from reef_perf.experiments.correlation.pinned import (
    KERNEL_DIR,
    PINNED_CONFIG,
    find_pinned_binary,
)
from reef_perf.experiments.matrix_diff.main import VME_CONFIG
from reef_perf.microbenchmarks import build_microbenchmark
from reef_perf.sim import run_sim

PROGRAMS = (
    "addr_forward",
    "alu_indep",
    "branch_group",
    "div_chain",
    "ext_store_stream",
    "fp_chain",
    "gemv_int8",
    "kernels/embedding_int8",
    "vec_memcpy",
    "frontend/jal_fwd",
    "vector_units/vadd_indep",
    "vector_units/vdiv_m8_indep",
    "vector_units/vdiv_m8_then_vadd_m8",
    "vector_units/vdiv_small_chain",
    "vector_units/vfadd_chain",
    "vector_units/vfdiv_indep",
    "lsu/vload_macc",
    "lsu/vstore_m4",
    "matrix/gemm_tile",
    "matrix/mac_rob",
    "matrix/tile_load32_burst",
    "matrix/move_vt_use",
    "matrix/tile_load_behind_mac",
    "matrix/vtdiscard_after_mac",
    "pinned/fdiv_indep",
    "pinned/fetch_jal_chain",
    "pinned/vsext_chain",
    "pinned/vwmacc_m2_chain",
)
"""Programs the switches are run on: microbenchmarks (built with ITERS 10
where they take it) and kernel ELFs (kernels/)."""

REQUIRES: dict[str, dict[str, Any]] = {
    # M3's fetch unit never has an instruction behind a jump in its cycle;
    # with the pinned one it does when dispatch is backed up (kernels' calls).
    "core.dispatch.params.jump_ends_group": {
        "frontend.params.fetch_reorder_buffer": True
    },
    "core.lsu.params.lsu_v3_tile": {"core.lsu.params.lsu_v3": True},
    "core.dispatch.params.load_waw_advance": {"core.lsu.params.lsu_v3": True},
    "vector.params.vec_store_completion_matches_pc": {
        "vector.params.vec_completion_matches_pc": False
    },
    "matrix.params.mtx_move_vector_floor": {
        "matrix.params.mtx_move_vector_latency": 1
    },
    "core.scalar_exec.params.fdiv_occupancy": {
        "core.scalar_exec.params.fdiv_latency": 13
    },
    "matrix.params.mtx_lsu_holds_full_fifo_beat": {
        "core.lsu.params.lsu_v3": True,
        "core.lsu.params.lsu_v3_tile": True,
    },
}
"""Switches that act only with others on, and those others' values."""

NO_EFFECT = {
    "frontend.params.zvt": "already on in m3_vme.yaml",
    "frontend.params.zvt_in_spike": "Spike executes Zvt with the same timing "
    "as reef-perf's layer "
    "(test_spike_and_layer_agree_on_every_zvt_instruction)",
    "core.dispatch.params.vector_scalar_waw_delay": "on the M3 machine no "
    "vector stream here is limited by its x-register writes",
}
"""Switches expected to leave every program's timing alone, and why."""


def _leaves(node: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    found = []
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else key
        found += (
            _leaves(value, path)
            if isinstance(value, dict)
            else [(path.removeprefix("top."), value)]
        )
    return found


SWITCHES = _leaves(yaml.safe_load(PINNED_CONFIG.read_text()))


def _param(value: bool | int | str) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


@pytest.fixture(scope="module")
def programs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The programs, assembled once."""
    out = tmp_path_factory.mktemp("switches")
    elfs: dict[str, Path] = {}
    for name in PROGRAMS:
        if name.startswith("kernels/"):
            kernel = name.removeprefix("kernels/")
            elfs[name] = KERNEL_DIR / "elfs" / f"{kernel}.elf"
            continue
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        elfs[name] = build_microbenchmark(name, out, {"ITERS": 10})
    return elfs


def _cycles(elfs: dict[str, Path], params: dict[str, Any]) -> dict[str, int]:
    cycles = {}
    for name, elf in elfs.items():
        result = run_sim(
            elf,
            overlay_configs=[VME_CONFIG],
            binary=find_pinned_binary(),
            params={f"top.{k}": _param(v) for k, v in params.items()},
        )
        assert result.instructions == result.functional_instructions, name
        cycles[name] = result.cycles
    return cycles


@pytest.mark.unit
def test_requirements_and_exceptions_name_switches() -> None:
    """REQUIRES and NO_EFFECT name switches of coral_pinned.yaml."""
    names = {name for name, _ in SWITCHES}
    assert set(REQUIRES) <= names
    assert set(NO_EFFECT) <= names
    for needed in REQUIRES.values():
        assert set(needed) <= names


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("switch", [name for name, _ in SWITCHES])
def test_switch_works_on_its_own(
    switch: str, cpp_build_dir: Path, programs: dict[str, Path]
) -> None:
    """The switch runs alone (with REQUIRES) and moves a program's timing."""
    del cpp_build_dir
    value = dict(SWITCHES)[switch]
    needed = REQUIRES.get(switch, {})
    base = _cycles(programs, needed)
    on = _cycles(programs, {**needed, switch: value})
    moved = sorted(name for name in on if on[name] != base[name])
    if switch in NO_EFFECT:
        assert not moved, f"{switch} moved {moved}; NO_EFFECT is stale"
    else:
        assert moved, f"{switch} alone moved no program"
