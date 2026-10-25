"""Collect one artifact: run one microbenchmark on the simulator.

Example:
    uv run python -m reef_perf.experiments.sweep.main \
        --microbenchmark gemv_int8 --param top.frontend.params.fetch_interval=2

The artifact is a JSON file holding the simulator's summary plus the
microbenchmark, simulation config and parameter overrides that produced it.
"""

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef_perf.microbenchmarks import build_microbenchmark, list_microbenchmarks
from reef_perf.sim import DEFAULT_SIM_CONFIG, MODULE_ROOT, run_sim

RESULTS_ROOT = MODULE_ROOT / "results" / "sweep"
"""Where the sweep experiment writes artifacts by default."""

ELF_DIR = MODULE_ROOT / "build" / "microbenchmarks"
"""Where assembled microbenchmarks are cached."""


def parse_params(items: list[str]) -> dict[str, str]:
    """Parse KEY=VALUE parameter overrides.

    Args:
        items: Strings such as "top.frontend.params.fetch_width=2".

    Returns:
        Mapping from parameter path to value.

    Raises:
        ValueError: If an item has no "=".
    """
    params = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        params[key] = value
    return params


def collect(
    microbenchmark: str,
    out_path: Path,
    sim_config: Path = DEFAULT_SIM_CONFIG,
    params: Mapping[str, str] | None = None,
    elf: Path | None = None,
) -> Path:
    """Run one microbenchmark and write its artifact.

    Args:
        microbenchmark: Microbenchmark name.
        out_path: JSON file to write.
        sim_config: Simulation config file.
        params: Parameter overrides.
        elf: Pre-built ELF; assembled into ELF_DIR when omitted.

    Returns:
        out_path.
    """
    elf_path = elf or build_microbenchmark(microbenchmark, ELF_DIR)
    result = run_sim(elf_path, sim_config=sim_config, params=params)
    artifact: dict[str, Any] = {
        "microbenchmark": microbenchmark,
        "sim_config": str(sim_config),
        "params": dict(params or {}),
        "result": result.to_json(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2) + "\n")
    return out_path


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments, or None for sys.argv.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--microbenchmark",
        required=True,
        choices=list_microbenchmarks(),
        help="microbenchmark",
    )
    parser.add_argument(
        "--sim-config",
        type=Path,
        default=DEFAULT_SIM_CONFIG,
        help="simulation config (default: configs/machines/m3.yaml)",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="parameter override; repeatable",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="artifact path (default: results/sweep/single/<name>.json)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Collect one artifact and print a one-line summary.

    Args:
        argv: Arguments, or None for sys.argv.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    out = args.out or RESULTS_ROOT / "single" / f"{args.microbenchmark}.json"
    path = collect(
        args.microbenchmark,
        out,
        sim_config=args.sim_config,
        params=parse_params(args.param),
    )
    result = json.loads(path.read_text())["result"]
    print(
        f"{args.microbenchmark}: {result['instructions']} instructions, "
        f"{result['cycles']} cycles, IPC {result['ipc']:.3f} -> {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
