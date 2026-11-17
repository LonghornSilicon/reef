"""Collect a sweep of artifacts described by a config file.

Example:
    uv run python -m reef_perf.experiments.sweep.sweep \
        --config python/reef_perf/experiments/sweep/sweeps/all.yaml

Every (variant, program) pair is one simulator run. A program is a
microbenchmark, assembled here, or a pre-built ELF named under `programs`.
Runs are independent processes and execute in parallel. Artifacts go to
results/sweep/<sweep name>/<variant>/<program>.json, with a summary.csv
alongside. The README describes the config format.
"""

import argparse
import csv
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from reef_perf.experiments.sweep.main import ELF_DIR, RESULTS_ROOT, collect
from reef_perf.microbenchmarks import (
    MICROBENCHMARK_DIR,
    build_microbenchmarks,
    list_microbenchmarks,
)
from reef_perf.sim import MODULE_ROOT, find_binary

OVERLAY_KEY = "overlay"
"""Variant and grid key naming config files rather than a parameter.

Parameter paths always contain a dot, so this name cannot clash with one.
"""

MATRIX_FIELDS = (
    "mac_utilization",
    "useful_macs",
    "capacity_macs",
    "array_busy_cycles",
    "mem_beats",
    "area_um2",
    "energy_pj",
)
"""Matrix-engine statistics copied into summary.csv, as matrix_<name>.

They cover the whole run, not only the ROI.
"""


@dataclass(frozen=True)
class SweepConfig:
    """A sweep, as described by a sweeps/*.yaml file.

    Attributes:
        name: Sweep name; artifacts go to results/sweep/<name>/.
        sim_config: Simulation config used for every run.
        microbenchmarks: Microbenchmark names, e.g. "alu_chain" or
            "matrix/mac_indep".
        variants: Variant name -> parameter overrides for that variant.
        overlay_configs: Config files applied after sim_config in every run.
        variant_overlays: Variant name -> config files applied after
            overlay_configs; variants without overlays are absent.
        programs: Program name -> pre-built ELF.
        sim_args: Further simulator options for every run, e.g.
            ["--ext-mem-mb", "160"].
        binary: Simulator executable name, e.g. "reef_perf_pinned" for
            coral_pinned.yaml; None runs reef_perf.
    """

    name: str
    sim_config: Path
    microbenchmarks: list[str]
    variants: dict[str, dict[str, str]]
    overlay_configs: tuple[Path, ...] = ()
    variant_overlays: dict[str, tuple[Path, ...]] = field(default_factory=dict)
    programs: dict[str, Path] = field(default_factory=dict)
    sim_args: tuple[str, ...] = ()
    binary: str | None = None

    def overlays(self, variant: str) -> tuple[Path, ...]:
        """Config files applied after sim_config for one variant.

        Args:
            variant: Variant name.

        Returns:
            The sweep's overlays, then the variant's.
        """
        return self.overlay_configs + self.variant_overlays.get(variant, ())

    def runs(self) -> list[tuple[str, str]]:
        """Every (variant, program) pair, in run order.

        Returns:
            The pairs; programs are the microbenchmarks, then `programs`.
        """
        names = self.microbenchmarks + list(self.programs)
        return [(variant, name) for variant in self.variants for name in names]


def _resolve(path: str) -> Path:
    """Expand environment variables; relative paths are from reef-perf/."""
    expanded = Path(os.path.expandvars(str(path)))
    return expanded if expanded.is_absolute() else MODULE_ROOT / expanded


def _param_str(value: object) -> str:
    """A parameter value as reef_perf's -p takes it (YAML true -> "true")."""
    return str(value).lower() if isinstance(value, bool) else str(value)


def _as_list(value: object) -> list[Any]:
    return list(value) if isinstance(value, list | tuple) else [value]


def _split_variant(
    name: str, settings: dict[str, Any] | None
) -> tuple[dict[str, str], tuple[Path, ...]]:
    """Separate a variant's parameter overrides from its config overlays."""
    settings = dict(settings or {})
    overlays = tuple(
        _resolve(path) for path in _as_list(settings.pop(OVERLAY_KEY, []))
    )
    for key in settings:
        if "." not in str(key):
            raise ValueError(
                f"variant {name!r}: {key!r} is not a parameter path "
                f"(top.<module>...) or {OVERLAY_KEY!r}"
            )
    return {str(k): _param_str(v) for k, v in settings.items()}, overlays


def _grid_label(key: str, value: object) -> str:
    if key == OVERLAY_KEY:
        return Path(str(value)).stem
    return f"{key.rsplit('.', 1)[-1]}-{_param_str(value)}"


def expand_grid(
    variants: dict[str, dict[str, Any] | None], grid: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Cross every variant with every combination of grid values.

    Args:
        variants: Variant name -> settings (parameter overrides, and
            optionally `overlay`). Empty means one unnamed base variant.
        grid: Parameter path (or `overlay`) -> list of values.

    Returns:
        Variant name -> settings. Names join the variant's name and one
        label per grid key, e.g. "coral_128pe__rob_entries-16". Grid
        overlays come after the variant's own.

    Raises:
        ValueError: If a grid entry has no values or two variants get the
            same name.
    """
    for key, values in grid.items():
        if not _as_list(values):
            raise ValueError(f"grid entry {key!r} has no values")
    base = variants or {"": {}}
    expanded: dict[str, dict[str, Any]] = {}
    keys = list(grid)
    for name, settings in base.items():
        for combo in itertools.product(*(_as_list(grid[k]) for k in keys)):
            merged = dict(settings or {})
            labels = [name] if name else []
            for key, value in zip(keys, combo, strict=True):
                labels.append(_grid_label(key, value))
                if key == OVERLAY_KEY:
                    merged[OVERLAY_KEY] = [
                        *_as_list(merged.get(OVERLAY_KEY, [])),
                        value,
                    ]
                else:
                    merged[key] = value
            label = "__".join(labels) or "default"
            if label in expanded:
                raise ValueError(f"two grid variants are named {label!r}")
            expanded[label] = merged
    return expanded


def load_config(path: Path) -> SweepConfig:
    """Read a sweep config file.

    Args:
        path: YAML file (see sweeps/all.yaml and the README for the format).

    Returns:
        The sweep.

    Raises:
        ValueError: If the config names an unknown microbenchmark, reuses a
            program name, or has a malformed variant or grid.
    """
    data = yaml.safe_load(Path(path).read_text())
    programs = {
        str(name): _resolve(elf)
        for name, elf in (data.get("programs") or {}).items()
    }
    # With only pre-built programs listed, run no microbenchmarks by default.
    microbenchmarks = data.get("microbenchmarks", [] if programs else "all")
    if microbenchmarks == "all":
        microbenchmarks = list_microbenchmarks()
    microbenchmarks = [str(name) for name in microbenchmarks or []]
    unknown = sorted(
        name
        for name in microbenchmarks
        if not (MICROBENCHMARK_DIR / f"{name}.S").is_file()
    )
    if unknown:
        raise ValueError(f"unknown microbenchmarks in {path}: {unknown}")
    reused = sorted(set(microbenchmarks) & set(programs))
    if reused:
        raise ValueError(f"program names reuse microbenchmarks: {reused}")

    sim_configs = [
        _resolve(cfg)
        for cfg in _as_list(data.get("sim_config", "configs/machines/m3.yaml"))
    ]
    variants = data.get("variants") or {}
    if data.get("grid"):
        variants = expand_grid(variants, data["grid"])
    variants = variants or {"default": {}}

    params: dict[str, dict[str, str]] = {}
    variant_overlays: dict[str, tuple[Path, ...]] = {}
    for name, settings in variants.items():
        params[str(name)], overlays = _split_variant(str(name), settings)
        if overlays:
            variant_overlays[str(name)] = overlays
    return SweepConfig(
        name=str(data["name"]),
        sim_config=sim_configs[0],
        microbenchmarks=microbenchmarks,
        variants=params,
        overlay_configs=tuple(sim_configs[1:]),
        variant_overlays=variant_overlays,
        programs=programs,
        sim_args=tuple(str(arg) for arg in data.get("sim_args") or []),
        binary=str(data["binary"]) if data.get("binary") else None,
    )


def artifact_path(sweep_dir: Path, variant: str, program: str) -> Path:
    """Where one run's artifact goes; "matrix/x" becomes "matrix-x.json".

    Args:
        sweep_dir: results/sweep/<sweep name>.
        variant: Variant name.
        program: Microbenchmark or program name.

    Returns:
        The artifact path.
    """
    return sweep_dir / variant / f"{program.replace('/', '-')}.json"


def run_sweep(
    config: SweepConfig,
    results_root: Path = RESULTS_ROOT,
    jobs: int | None = None,
) -> list[Path]:
    """Collect every artifact in a sweep, in parallel.

    Args:
        config: The sweep.
        results_root: Directory under which results/<name>/ is created.
        jobs: Parallel simulator processes; one per CPU by default.

    Returns:
        Paths of the artifacts written.

    Raises:
        FileNotFoundError: If a pre-built program's ELF or a config file is
            missing; checked before any run starts.
    """
    missing = [
        f"{name}: {elf}"
        for name, elf in config.programs.items()
        if not elf.is_file()
    ] + [
        str(cfg)
        for cfg in {config.sim_config}.union(
            *(config.overlays(v) for v in config.variants)
        )
        if not cfg.is_file()
    ]
    if missing:
        raise FileNotFoundError("missing files:\n  " + "\n  ".join(missing))

    for name in config.microbenchmarks:
        (ELF_DIR / name).parent.mkdir(parents=True, exist_ok=True)
    elfs = build_microbenchmarks(config.microbenchmarks, ELF_DIR)
    elfs.update(config.programs)
    binary = find_binary(config.binary) if config.binary else None
    sweep_dir = results_root / config.name
    with ThreadPoolExecutor(max_workers=jobs or os.cpu_count()) as pool:
        futures = [
            pool.submit(
                collect,
                program,
                artifact_path(sweep_dir, variant, program),
                config.sim_config,
                config.variants[variant],
                elfs[program],
                config.overlays(variant),
                config.sim_args,
                binary,
            )
            for variant, program in config.runs()
        ]
        paths = [future.result() for future in futures]
    write_summary(paths, sweep_dir / "summary.csv")
    return paths


def write_summary(artifacts: list[Path], out_path: Path) -> None:
    """Write one CSV row per artifact.

    The first five columns are the original layout; the ROI, matrix,
    overlay and parameter columns follow them.

    Args:
        artifacts: Artifact JSON files.
        out_path: CSV file to write.
    """
    loaded = [(path, json.loads(path.read_text())) for path in artifacts]
    param_keys = sorted({k for _, a in loaded for k in a.get("params", {})})
    fields = ["variant", "microbenchmark", "instructions", "cycles", "ipc"]
    fields += ["roi_cycles", "roi_instructions"]
    fields += [f"matrix_{name}" for name in MATRIX_FIELDS]
    fields += ["overlay_configs", *param_keys]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for path, artifact in loaded:
            result = artifact["result"]
            matrix = result.get("matrix") or {}
            row = {
                "variant": path.parent.name,
                "microbenchmark": artifact["microbenchmark"],
                "instructions": result["instructions"],
                "cycles": result["cycles"],
                "ipc": f"{result['ipc']:.4f}",
                "roi_cycles": result.get("roi_cycles"),
                "roi_instructions": result.get("roi_instructions"),
                "overlay_configs": " ".join(
                    Path(cfg).stem
                    for cfg in artifact.get("overlay_configs", [])
                ),
            }
            row.update(
                {f"matrix_{name}": matrix.get(name) for name in MATRIX_FIELDS}
            )
            row.update(artifact.get("params", {}))
            writer.writerow(row)


def describe(config: SweepConfig) -> str:
    """List a sweep's runs without running them.

    Args:
        config: The sweep.

    Returns:
        One line per variant (its overlays and overrides), then the programs.
    """
    lines = [f"sweep {config.name}: base {config.sim_config.name}"]
    for variant in config.variants:
        overlays = " ".join(cfg.name for cfg in config.overlays(variant))
        params = " ".join(
            f"{k}={v}" for k, v in config.variants[variant].items()
        )
        lines.append(f"  {variant}: {overlays} {params}".rstrip())
    names = config.microbenchmarks + list(config.programs)
    lines.append(f"programs: {' '.join(names)}")
    if config.sim_args:
        lines.append(f"simulator options: {' '.join(config.sim_args)}")
    if config.binary:
        lines.append(f"simulator: {config.binary}")
    lines.append(f"{len(config.runs())} runs")
    return "\n".join(lines)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments, or None for sys.argv.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, required=True, help="sweep config"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_ROOT,
        help="results root (default: results/sweep)",
    )
    parser.add_argument(
        "--jobs", type=int, default=None, help="parallel runs (default: CPUs)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the variants and programs without running them",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run a sweep and print a table.

    Args:
        argv: Arguments, or None for sys.argv.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    config = load_config(args.config)
    if args.list:
        print(describe(config))
        return 0
    paths = run_sweep(config, args.out, args.jobs)
    rows = []
    for path in paths:
        artifact = json.loads(path.read_text())
        result = artifact["result"]
        rows.append(
            (
                path.parent.name,
                artifact["microbenchmark"],
                result["instructions"],
                result["cycles"],
                "" if result["roi_cycles"] is None else result["roi_cycles"],
                f"{result['ipc']:.3f}",
            )
        )
    width_v = max([10, *(len(row[0]) for row in rows)])
    width_p = max([14, *(len(row[1]) for row in rows)])
    print(
        f"{'variant':<{width_v}} {'microbenchmark':<{width_p}} "
        f"{'insts':>10} {'cycles':>10} {'roi_cycles':>10} IPC"
    )
    for variant, name, insts, cycles, roi, ipc in rows:
        print(
            f"{variant:<{width_v}} {name:<{width_p}} {insts:>10} "
            f"{cycles:>10} {roi:>10} {ipc}"
        )
    print(f"\n{len(paths)} artifacts in {args.out / config.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
