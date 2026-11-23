"""Sweep config parsing and run setup, with a stand-in simulator."""

import csv
import json
import stat
import sys
from pathlib import Path

import pytest

from reef_perf.experiments.sweep.sweep import (
    describe,
    expand_grid,
    load_config,
    run_sweep,
)
from reef_perf.sim import BIN_DIR_ENV, MODULE_ROOT

pytestmark = pytest.mark.unit

SWEEPS = MODULE_ROOT / "python/reef_perf/experiments/sweep/sweeps"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "sweep.yaml"
    path.write_text(text)
    return path


def test_all_yaml_keeps_its_meaning() -> None:
    """The original single-config format loads as before."""
    config = load_config(SWEEPS / "all.yaml")
    assert config.sim_config == MODULE_ROOT / "configs/machines/m3.yaml"
    assert config.variants == {"m3": {}}
    assert config.overlay_configs == ()
    assert config.overlays("m3") == ()
    assert config.programs == {}
    assert config.sim_args == ()


def test_matrix_presets_names_variants_by_preset() -> None:
    """A grid over overlay files gives one variant per preset."""
    config = load_config(SWEEPS / "matrix_presets.yaml")
    assert list(config.variants) == [
        "coral_vme",
        "coral_128pe",
        "os_systolic_16x16",
        "ws_systolic_16x16",
        "tree_8x8",
    ]
    assert config.overlays("coral_128pe") == (
        MODULE_ROOT / "configs/machines/m3_vme.yaml",
        MODULE_ROOT / "configs/matrix/coral_128pe.yaml",
    )
    assert "matrix/mac_indep" in config.microbenchmarks
    for variant in config.variants:
        assert all(path.is_file() for path in config.overlays(variant))


def test_e2e_kernels_crosses_variants_with_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Named variants times grid values, with environment-expanded ELFs."""
    monkeypatch.setenv("E2E_ROOT", "/kernels")
    config = load_config(SWEEPS / "e2e_kernels.yaml")
    assert len(config.variants) == 8
    name = "coral_128pe__rob_entries-16__lsu_tile_cost-true"
    assert config.variants[name] == {
        "top.core.rob.params.rob_entries": "16",
        "top.core.lsu.params.lsu_tile_cost": "true",
    }
    assert config.overlays(name)[-1] == (
        MODULE_ROOT / "configs/matrix/coral_128pe.yaml"
    )
    assert config.overlays("coral__rob_entries-8__lsu_tile_cost-false")[-1] == (
        MODULE_ROOT / "configs/memory/psram_kv.yaml"
    )
    assert config.microbenchmarks == []
    assert config.programs["fc1_t1"] == Path("/kernels/t1/fc1.elf")
    assert config.sim_args == ("--ext-mem-mb", "160")
    assert describe(config).endswith("64 runs")


def test_e2e_firmware_names_its_machine_and_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The firmware sweeps differ by machine; pinned names its binary."""
    monkeypatch.setenv("E2E_FW_ROOT", "/fw")
    m3 = load_config(SWEEPS / "e2e_firmware_m3_vme.yaml")
    pinned = load_config(SWEEPS / "e2e_firmware_pinned.yaml")
    assert m3.binary is None
    assert pinned.binary == "reef_perf_pinned"
    assert pinned.overlay_configs == (
        MODULE_ROOT / "configs/machines/coral_pinned.yaml",
    )
    assert m3.programs == pinned.programs
    assert pinned.programs["e2e_nomatrix"] == Path("/fw/e2e_nomatrix.elf")
    assert "simulator: reef_perf_pinned" in describe(pinned)
    assert describe(pinned).endswith("18 runs")


def test_grid_without_variants_and_duplicate_names() -> None:
    """Grid-only sweeps are named by their labels; clashes are errors."""
    grid = {"top.core.rob.params.rob_entries": [8, 16]}
    assert list(expand_grid({}, grid)) == ["rob_entries-8", "rob_entries-16"]
    with pytest.raises(ValueError, match="named"):
        expand_grid({}, {"top.a.x": [1, 1]})
    with pytest.raises(ValueError, match="no values"):
        expand_grid({}, {"top.a.x": []})


def test_bad_configs_fail_by_name(tmp_path: Path) -> None:
    """Unknown microbenchmarks and non-parameter keys are reported."""
    with pytest.raises(ValueError, match="no_such"):
        load_config(_write(tmp_path, "name: x\nmicrobenchmarks: [no_such]\n"))
    with pytest.raises(ValueError, match="rob_entries"):
        load_config(
            _write(tmp_path, "name: x\nvariants:\n  a: {rob_entries: 8}\n")
        )


FAKE_SIM = """#!{python}
import json, sys
args = sys.argv[1:]
out = args[args.index("--json") + 1]
with open("{log}/argv.json", "w") as f:
    json.dump(args, f)
result = {{
    "program": args[args.index("--elf") + 1], "instructions": 100,
    "functional_instructions": 100, "cycles": 200, "ipc": 0.5,
    "taken_redirects": 0, "mcycle_reads": [10, 60], "roi_cycles": 50,
    "roi_instructions": 20, "dispatch": {{}}, "pools": {{}}, "topdown": {{}},
    "matrix": {{"mac_utilization": 0.25, "useful_macs": 4096}},
}}
with open(out, "w") as f:
    json.dump(result, f)
"""


def test_run_sweep_passes_configs_options_and_programs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overlays, -p overrides and options reach the simulator in order."""
    bin_dir, log = tmp_path / "bin", tmp_path / "log"
    bin_dir.mkdir()
    log.mkdir()
    sim = bin_dir / "reef_perf"
    sim.write_text(FAKE_SIM.format(python=sys.executable, log=log))
    sim.chmod(sim.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv(BIN_DIR_ENV, str(bin_dir))
    elf = tmp_path / "k.elf"
    elf.write_bytes(b"")
    config = load_config(
        _write(
            tmp_path,
            f"""
name: t
sim_config: [configs/machines/m3.yaml, configs/machines/m3_vme.yaml]
sim_args: [--ext-mem-mb, "160"]
programs: {{kernel: {elf}}}
grid:
  overlay: [configs/matrix/coral_128pe.yaml]
  top.core.rob.params.rob_entries: [16]
""",
        )
    )
    paths = run_sweep(config, results_root=tmp_path / "out", jobs=1)
    assert [p.relative_to(tmp_path / "out") for p in paths] == [
        Path("t/coral_128pe__rob_entries-16/kernel.json")
    ]

    args = json.loads((log / "argv.json").read_text())
    configs = [Path(args[i + 1]).name for i, a in enumerate(args) if a == "-c"]
    assert configs == ["m3.yaml", "m3_vme.yaml", "coral_128pe.yaml"]
    assert args[args.index("-p") + 1 : args.index("-p") + 3] == [
        "top.core.rob.params.rob_entries",
        "16",
    ]
    assert args[-2:] == ["--ext-mem-mb", "160"]

    with (tmp_path / "out/t/summary.csv").open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames is not None
    assert reader.fieldnames[:5] == [
        "variant",
        "microbenchmark",
        "instructions",
        "cycles",
        "ipc",
    ]
    (row,) = rows
    assert row["roi_cycles"] == "50"
    assert row["matrix_mac_utilization"] == "0.25"
    assert row["matrix_area_um2"] == ""
    assert row["overlay_configs"] == "m3_vme coral_128pe"
    assert row["top.core.rob.params.rob_entries"] == "16"


def test_missing_program_fails_before_running(tmp_path: Path) -> None:
    """A missing ELF stops the sweep with its name and path."""
    config = load_config(
        _write(tmp_path, "name: t\nprograms: {gone: /nowhere/x.elf}\n")
    )
    with pytest.raises(FileNotFoundError, match=r"gone: /nowhere/x\.elf"):
        run_sweep(config, results_root=tmp_path)
