"""Plot a sweep collected by run.py.

Example:
    uv run python -m reef_perf.experiments.run.plot \
        --config src/reef_perf/experiments/run/configs/all.yaml

Writes ipc.png (IPC per workload, one bar per variant) and stalls.png
(where each workload's cycles went) next to the sweep's artifacts.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

from reef_perf.experiments.run.main import RESULTS_ROOT
from reef_perf.experiments.run.run import load_config

Artifacts = dict[str, dict[str, dict[str, Any]]]
"""Variant -> workload -> artifact."""


def load_artifacts(sweep_dir: Path) -> Artifacts:
    """Read every artifact of a sweep.

    Args:
        sweep_dir: results/run/<sweep name>.

    Returns:
        Artifacts keyed by variant, then workload.

    Raises:
        FileNotFoundError: If the sweep has not been run.
    """
    artifacts: Artifacts = {}
    for path in sorted(sweep_dir.glob("*/*.json")):
        artifact = json.loads(path.read_text())
        artifacts.setdefault(path.parent.name, {})[artifact["workload"]] = (
            artifact
        )
    if not artifacts:
        raise FileNotFoundError(f"no artifacts in {sweep_dir}; run run.py")
    return artifacts


def plot_ipc(artifacts: Artifacts, out_path: Path) -> None:
    """Bar chart of IPC per workload, one bar per variant.

    Args:
        artifacts: Sweep artifacts.
        out_path: PNG to write.
    """
    workloads = sorted({w for runs in artifacts.values() for w in runs})
    width = 0.8 / len(artifacts)
    fig = Figure(figsize=(max(6, len(workloads) * 0.7), 4))
    ax = fig.subplots()
    for i, (variant, runs) in enumerate(sorted(artifacts.items())):
        xs = [j + i * width for j in range(len(workloads))]
        ys = [runs[w]["result"]["ipc"] if w in runs else 0 for w in workloads]
        ax.bar(xs, ys, width, label=variant)
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(workloads))])
    ax.set_xticklabels(workloads, rotation=45, ha="right")
    ax.set_ylabel("IPC")
    ax.set_title("IPC per workload")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)


def plot_stalls(artifacts: Artifacts, out_path: Path) -> None:
    """Stacked bars: fraction of cycles dispatching vs each stall reason.

    One subplot per variant.

    Args:
        artifacts: Sweep artifacts.
        out_path: PNG to write.
    """
    variants = sorted(artifacts)
    fig = Figure(figsize=(10, 3.5 * len(variants)))
    axes = fig.subplots(len(variants), 1, squeeze=False)
    for ax, variant in zip(axes[:, 0], variants, strict=True):
        runs = artifacts[variant]
        workloads = sorted(runs)
        stall_keys = sorted(
            key
            for key in runs[workloads[0]]["result"]["dispatch"]
            if key.startswith("stall_")
        )
        categories = ["cycles_with_dispatch", *stall_keys]
        bottoms = [0.0] * len(workloads)
        for category in categories:
            fractions = []
            for workload in workloads:
                result = runs[workload]["result"]
                cycles = result["cycles"] or 1
                fractions.append(result["dispatch"][category] / cycles)
            label = category.removeprefix("stall_")
            ax.bar(workloads, fractions, bottom=bottoms, label=label)
            bottoms = [b + f for b, f in zip(bottoms, fractions, strict=True)]
        ax.set_ylabel("fraction of cycles")
        ax.set_title(f"Dispatch breakdown ({variant})")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize="small", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)


def plot_sweep(sweep_dir: Path) -> list[Path]:
    """Write all plots for a sweep.

    Args:
        sweep_dir: results/run/<sweep name>.

    Returns:
        Paths of the PNGs written.
    """
    artifacts = load_artifacts(sweep_dir)
    outputs = [sweep_dir / "ipc.png", sweep_dir / "stalls.png"]
    plot_ipc(artifacts, outputs[0])
    plot_stalls(artifacts, outputs[1])
    return outputs


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
        help="results root (default: results/run)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Plot a sweep.

    Args:
        argv: Arguments, or None for sys.argv.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    config = load_config(args.config)
    for path in plot_sweep(args.out / config.name):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
