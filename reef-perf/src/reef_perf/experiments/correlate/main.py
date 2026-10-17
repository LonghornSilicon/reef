
"""Compare one REEF run artifact against M3 RTL measurements."""

import csv
import json
from pathlib import Path

RTL_CSV = Path(__file__).parent / "data" / "rtl_m3.csv"


def load_rtl_cycles(workload: str, csv_path: Path = RTL_CSV) -> int:
    """Return the RTL ROI cycle count for a workload."""
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["workload"] == workload:
                return int(row["rtl_roi_cycles"])
    raise ValueError(f"No RTL measurements for workload: {workload}")


def correlate(run_artifact: Path, out_path: Path) -> Path:
    """Compare one REEF run against RTL and save the correlation artifact."""
    artifact = json.loads(run_artifact.read_text())
    workload = artifact["workload"]
    result = artifact["result"]

    if not result.get("roi_complete", False):
        raise ValueError(f"ROI was not completed for {workload}")

    rtl_cycles = load_rtl_cycles(workload)
    model_cycles = int(result["roi_cycles"])

    error_pct = (model_cycles - rtl_cycles) / rtl_cycles * 100

    dispatch = result.get("dispatch", {})

    stalls = {
        key.removeprefix("stall_"): value
        for key, value in dispatch.items()
        if key.startswith("stall_")
    }

    top_stall = max(stalls, key=stalls.get) if stalls else "none"

    correlation = {
        "workload": workload,
        "rtl_cycles": rtl_cycles,
        "model_cycles": model_cycles,
        "error_pct": error_pct,
        "model_top_stall": top_stall,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(correlation, indent=2) + "\n")
    return out_path
