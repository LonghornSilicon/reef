"""Collect a sweep of artifacts described by a config file.

Example:
    uv run python -m reef_perf.experiments.sweep.sweep \
        --config python/reef_perf/experiments/sweep/sweeps/all.yaml

Every (variant, microbenchmark) pair is one simulator run. Runs are independent
processes and execute in parallel. Artifacts go to
results/sweep/<sweep name>/<variant>/<microbenchmark>.json, with a summary.csv
alongside.
"""

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import yaml

from reef_perf.experiments.sweep.main import ELF_DIR, RESULTS_ROOT, collect
from reef_perf.microbenchmarks import (
    build_microbenchmarks,
    list_microbenchmarks,
)
from reef_perf.sim import MODULE_ROOT


@dataclass(frozen=True)
class SweepConfig:
    """A sweep, as described by a sweeps/*.yaml file.

    Attributes:
        name: Sweep name; artifacts go to results/sweep/<name>/.
        sim_config: Simulation config used for every run.
        microbenchmarks: Microbenchmark names.
        variants: Variant name -> parameter overrides for that variant.
    """

    name: str
    sim_config: Path
    microbenchmarks: list[str]
    variants: dict[str, dict[str, str]]


def load_config(path: Path) -> SweepConfig:
    """Read a sweep config file.

    Args:
        path: YAML file (see sweeps/all.yaml for the format).

    Returns:
        The sweep.

    Raises:
        ValueError: If the config names an unknown microbenchmark.
    """
    data = yaml.safe_load(Path(path).read_text())
    microbenchmarks = data.get("microbenchmarks", "all")
    available = list_microbenchmarks()
    if microbenchmarks == "all":
        microbenchmarks = available
    unknown = sorted(set(microbenchmarks) - set(available))
    if unknown:
        raise ValueError(f"unknown microbenchmarks in {path}: {unknown}")
    variants = data.get("variants") or {"default": {}}
    return SweepConfig(
        name=str(data["name"]),
        sim_config=MODULE_ROOT
        / data.get("sim_config", "configs/machines/m3.yaml"),
        microbenchmarks=list(microbenchmarks),
        variants={
            str(name): {str(k): str(v) for k, v in (params or {}).items()}
            for name, params in variants.items()
        },
    )


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
    """
    elfs = build_microbenchmarks(config.microbenchmarks, ELF_DIR)
    sweep_dir = results_root / config.name
    tasks = [
        (variant, microbenchmark, params)
        for variant, params in config.variants.items()
        for microbenchmark in config.microbenchmarks
    ]
    with ThreadPoolExecutor(max_workers=jobs or os.cpu_count()) as pool:
        futures = [
            pool.submit(
                collect,
                microbenchmark,
                sweep_dir / variant / f"{microbenchmark}.json",
                config.sim_config,
                params,
                elfs[microbenchmark],
            )
            for variant, microbenchmark, params in tasks
        ]
        paths = [future.result() for future in futures]
    write_summary(paths, sweep_dir / "summary.csv")
    return paths


def write_summary(artifacts: list[Path], out_path: Path) -> None:
    """Write one CSV row per artifact.

    Args:
        artifacts: Artifact JSON files.
        out_path: CSV file to write.
    """
    fields = ["variant", "microbenchmark", "instructions", "cycles", "ipc"]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for path in artifacts:
            artifact = json.loads(path.read_text())
            result = artifact["result"]
            writer.writerow(
                {
                    "variant": path.parent.name,
                    "microbenchmark": artifact["microbenchmark"],
                    "instructions": result["instructions"],
                    "cycles": result["cycles"],
                    "ipc": f"{result['ipc']:.4f}",
                }
            )


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
    paths = run_sweep(config, args.out, args.jobs)
    print(
        f"{'variant':<10} {'microbenchmark':<14} {'insts':>10} "
        f"{'cycles':>10} IPC"
    )
    for path in paths:
        artifact = json.loads(path.read_text())
        result = artifact["result"]
        print(
            f"{path.parent.name:<10} {artifact['microbenchmark']:<14} "
            f"{result['instructions']:>10} {result['cycles']:>10} "
            f"{result['ipc']:.3f}"
        )
    print(f"\n{len(paths)} artifacts in {args.out / config.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
