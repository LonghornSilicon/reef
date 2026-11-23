"""E6.1/E6.2: the epilogue's share of fc1, from the matrix_e6_epilogue sweep.

Usage:
    python epilogue_share.py results/sweep/matrix_e6_epilogue

The fc1_epi_* builds read mcycle just before and after each epilogue (four
epilogue calls per 64-column group, 32 groups for fc1). Their cycles are the
sum over those paired reads (every mcycle read but the ROI's first and last);
the share divides that by roi_cycles. The plain fc1 runs of the same variant
give the ROI without the extra reads, to show they do not move it. Writes
epilogue_share.json next to the sweep's artifacts and prints where it went.
"""

import json
import sys
from pathlib import Path


def epilogue_rows(sweep_dir: Path) -> list[dict[str, object]]:
    """One row per instrumented run.

    Args:
        sweep_dir: results/sweep/matrix_e6_epilogue.

    Returns:
        Rows with the epilogue cycles, the ROI and the share.

    Raises:
        ValueError: If a run's mcycle reads do not pair up.
    """
    rows = []
    for path in sorted(sweep_dir.glob("*/fc1_epi_*.json")):
        result = json.loads(path.read_text())["result"]
        inner = result["mcycle_reads"][1:-1]
        if not inner or len(inner) % 2:
            raise ValueError(f"{path}: {len(inner)} inner mcycle reads")
        cycles = sum(
            b - a for a, b in zip(inner[::2], inner[1::2], strict=True)
        )
        plain = path.with_name(path.name.replace("fc1_epi_", "fc1_"))
        plain_roi = json.loads(plain.read_text())["result"]["roi_cycles"]
        rows.append(
            {
                "variant": path.parent.name,
                "program": path.stem,
                "epilogues": len(inner) // 2,
                "epilogue_cycles": cycles,
                "roi_cycles": result["roi_cycles"],
                "epilogue_share": cycles / result["roi_cycles"],
                "plain_roi_cycles": plain_roi,
                "instrumentation_roi_change": result["roi_cycles"] / plain_roi
                - 1,
            }
        )
    return rows


def main(argv: list[str]) -> int:
    """Write epilogue_share.json for one sweep directory.

    Args:
        argv: The sweep directory.

    Returns:
        Process exit code.
    """
    if len(argv) != 1:
        print(__doc__)
        return 2
    sweep_dir = Path(argv[0])
    rows = epilogue_rows(sweep_dir)
    out = sweep_dir / "epilogue_share.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {out} ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
