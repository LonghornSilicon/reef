"""Command line: `python -m reef_perf.vector_view TRACE.tsv [-o OUT.html]`."""

import argparse
from pathlib import Path

from reef_perf.sim import DEFAULT_SIM_CONFIG
from reef_perf.vector_view.build import (
    load_params,
    parse_vector_run,
    render_html,
)


def main() -> None:
    """Writes the vector view for a trace file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="file from reef_perf --trace")
    parser.add_argument("-o", "--output", type=Path, help="default: TRACE.html")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_SIM_CONFIG,
        help="machine config the trace was run with (default: %(default)s)",
    )
    args = parser.parse_args()

    run = parse_vector_run(args.trace, load_params(args.config))
    out = args.output or args.trace.with_suffix(".html")
    out.write_text(render_html(run, args.trace.stem))
    print(
        f"Wrote {out} ({len(run.insts)} vector instructions, "
        f"{run.num_unverified} unverified)"
    )


if __name__ == "__main__":
    main()
