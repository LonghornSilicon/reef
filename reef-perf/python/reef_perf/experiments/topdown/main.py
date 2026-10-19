"""Top-down analysis of microbenchmarks: where the dispatch slots went.

Example:
    uv run python -m reef_perf.experiments.topdown.main
    uv run python -m reef_perf.experiments.topdown.main \
        --microbenchmark load_chain --param top.memory.params.dtcm_latency=3

Runs the chosen microbenchmarks (all by default) and writes, under
results/topdown/: one tree per microbenchmark (<name>.json), a level-2 table
(summary.md) and a stacked bar chart (topdown.png). With one microbenchmark it
also prints the full tree. See docs/topdown.md for the method.
"""

import argparse
import json
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from matplotlib.figure import Figure

from reef_perf.experiments.sweep.main import ELF_DIR, parse_params
from reef_perf.microbenchmarks import (
    build_microbenchmarks,
    list_microbenchmarks,
)
from reef_perf.sim import DEFAULT_SIM_CONFIG, MODULE_ROOT, run_sim
from reef_perf.topdown import Node, build_tree, render

RESULTS_ROOT = MODULE_ROOT / "results" / "topdown"
"""Where the top-down experiment writes artifacts by default."""

LEVEL2 = [
    "retiring",
    "frontend_bound.branch_resteer",
    "frontend_bound.fetch_bandwidth",
    "backend_bound.core",
    "backend_bound.memory",
    "backend_bound.vector",
    "backend_bound.matrix",
]
"""Level-2 categories, in table and chart order."""


def analyze(
    names: list[str],
    sim_config: Path = DEFAULT_SIM_CONFIG,
    params: Mapping[str, str] | None = None,
    jobs: int | None = None,
) -> dict[str, Node]:
    """Run microbenchmarks and build their top-down trees.

    Args:
        names: Microbenchmarks to run.
        sim_config: Simulation config.
        params: Parameter overrides.
        jobs: Parallel simulator processes; one per CPU by default.

    Returns:
        Tree per microbenchmark, in the order given.
    """
    elfs = build_microbenchmarks(names, ELF_DIR)
    with ThreadPoolExecutor(max_workers=jobs or os.cpu_count()) as pool:
        results = list(
            pool.map(
                lambda n: run_sim(
                    elfs[n], sim_config=sim_config, params=params
                ),
                names,
            )
        )
    return {
        name: build_tree(result.topdown)
        for name, result in zip(names, results, strict=True)
    }


def level2_shares(tree: Node) -> dict[str, float]:
    """Share of all slots in each level-2 category.

    Args:
        tree: One microbenchmark's tree.

    Returns:
        Category -> fraction of slots, for every entry of LEVEL2.
    """
    slots = tree.level(2)
    return {
        category: slots.get(category, 0) / tree.slots if tree.slots else 0.0
        for category in LEVEL2
    }


def render_table(trees: Mapping[str, Node]) -> str:
    """Render the level-2 shares of every microbenchmark as Markdown.

    Args:
        trees: Tree per microbenchmark.

    Returns:
        The table.
    """
    lines = [
        "| Microbenchmark | " + " | ".join(LEVEL2) + " |",
        "| --- |" + " ---: |" * len(LEVEL2),
    ]
    for name, tree in trees.items():
        shares = level2_shares(tree)
        cells = " | ".join(f"{100 * shares[c]:.1f}%" for c in LEVEL2)
        lines.append(f"| `{name}` | {cells} |")
    return "\n".join(lines) + "\n"


def plot(trees: Mapping[str, Node], out_path: Path) -> Path:
    """Stacked bar chart of the level-2 shares, one bar per microbenchmark.

    Args:
        trees: Tree per microbenchmark.
        out_path: PNG to write.

    Returns:
        out_path.
    """
    names = list(trees)
    fig = Figure(figsize=(8, max(3, 0.4 * len(names) + 1)))
    ax = fig.subplots()
    left = [0.0] * len(names)
    for category in LEVEL2:
        widths = [100 * level2_shares(trees[n])[category] for n in names]
        ax.barh(names, widths, left=left, label=category)
        left = [a + b for a, b in zip(left, widths, strict=True)]
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of dispatch slots")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    fig.tight_layout()
    fig.savefig(out_path)
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
        action="append",
        choices=list_microbenchmarks(),
        help="microbenchmark to analyze; repeatable (default: all)",
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
        default=RESULTS_ROOT,
        help="output directory (default: results/topdown)",
    )
    parser.add_argument(
        "--jobs", type=int, default=None, help="parallel runs (default: CPUs)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Analyze, write the artifacts and print the results.

    Args:
        argv: Arguments, or None for sys.argv.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    names = args.microbenchmark or list_microbenchmarks()
    trees = analyze(names, args.sim_config, parse_params(args.param), args.jobs)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, tree in trees.items():
        (args.out / f"{name}.json").write_text(
            json.dumps(asdict(tree), indent=2) + "\n"
        )
    table = render_table(trees)
    (args.out / "summary.md").write_text(table)
    plot(trees, args.out / "topdown.png")
    if len(trees) == 1:
        print(render(next(iter(trees.values()))))
    else:
        print(table)
    print(f"Artifacts in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
