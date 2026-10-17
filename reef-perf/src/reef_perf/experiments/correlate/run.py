
"""Correlate REEF simulation results against M3 RTL measurements."""

import argparse
import json
from pathlib import Path

from reef_perf.experiments.correlate.main import correlate
from reef_perf.experiments.run.run import load_config, run_sweep
from reef_perf.sim import MODULE_ROOT

RESULTS_ROOT = MODULE_ROOT / "results" / "correlate"


def run_correlation(config_path: Path) -> list[dict]:
    """Run a workload sweep and correlate every result against RTL."""
    config = load_config(config_path)
    run_paths = run_sweep(config)

    output_dir = RESULTS_ROOT / config.name
    results = []

    for run_path in run_paths:
        out_path = (
            output_dir / run_path.parent.name / run_path.name
        )
        correlate(run_path, out_path)
        results.append(json.loads(out_path.read_text()))

    return results


def main() -> int:
    """Run the correlation experiment and print its summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    results = run_correlation(args.config)

    print(
        f"{'workload':<18} {'rtl_cycles':>12} "
        f"{'model_cycles':>14} {'error%':>10}  model top stall"
    )

    for result in results:
        print(
            f"{result['workload']:<18} "
            f"{result['rtl_cycles']:>12} "
            f"{result['model_cycles']:>14} "
            f"{result['error_pct']:>9.2f}%  "
            f"{result['model_top_stall']}"
        )

    mean_absolute_error = (
        sum(abs(result["error_pct"]) for result in results) / len(results)
    )

    print(f"\nMean absolute error: {mean_absolute_error:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
