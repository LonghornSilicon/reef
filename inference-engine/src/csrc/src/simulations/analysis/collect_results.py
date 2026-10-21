"""Collect the sim test outputs into out/results.json; optionally log them.

    uv run --group sim python analysis/collect_results.py   (from simulations/)
    uv run --group sim python analysis/collect_results.py --log "what changed"

Reads every call record `sim.py test` left in out/<test>/*.npz (cycles,
host write cost and the test's own error check), derives the VME cycles per
matmul_step from the k sweep, and writes out/results.json, which
roofline.py reads. --log appends one row of the headline metrics (METRICS)
to the performance log in analysis/RESULTS.md, with the current commit, the
note, and the change against the previous row.
"""

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIMS = HERE.parent
OUT = SIMS / "out"
LOG = HERE / "RESULTS.md"
HARDWARE = json.loads((HERE / "hardware.json").read_text())

# Headline metrics in the RESULTS.md log: column -> sim job, attention
# stream (sum of its blocks and finish), or microbench key. RESULTS.md says
# what each one measures; keep the two in sync.
METRICS = {
    "vme_matmul": ("job", "vme_matmul_u8_16x512x16"),
    "rvv_matmul": ("job", "matmul_rvv_1x512x32"),
    "vme_step": ("micro", "vme_cycles_per_matmul_step"),
    "attn_prefill": ("stream", "vme_prefill"),
    "attn_decode": ("stream", "vme_decode"),
    "attn_d32_prefill": ("stream", "vme_d32_prefill"),
    "attn_d32_decode": ("stream", "vme_d32_decode"),
    "attn_d32_local": ("stream", "vme_d32_local"),
    "lut_table": ("job", "softmax_lut_table"),
    "gelu": ("job", "gelu"),
    "layernorm": ("job", "layernorm"),
    "residual": ("job", "residual_add"),
}
CHANGE_THRESHOLD = 0.01  # moves smaller than 1% are noise in the log


def load_jobs() -> list[dict]:
    """Load every saved call record."""
    jobs = []
    for path in sorted(OUT.glob("*/*.npz")):
        data = dict(np.load(path))
        job = json.loads(str(data.pop("meta")))
        job["out_bytes"] = sum(
            v.nbytes for k, v in data.items() if k.startswith("out_")
        )
        jobs.append(job)
    return jobs


def microbench(jobs: list[dict]) -> dict:
    """Derive the VME step cost from the k sweep."""
    out = {}
    # cycles vs k at one 16x16 tile: the slope is cycles per matmul_step
    sweep = sorted(
        (j["shape"][1], j["cycles"])
        for j in jobs
        if j["name"].startswith("sweep_vme_k")
    )
    if len(sweep) >= 2:
        steps = np.array([k / 4 for k, _ in sweep])
        slope, intercept = np.polyfit(steps, [c for _, c in sweep], 1)
        out["vme_cycles_per_matmul_step"] = round(float(slope), 2)
        out["vme_tile_overhead_cycles"] = round(float(intercept), 1)
        out["vme_peak_macs_per_cycle"] = round(1024 / float(slope), 1)
    return out


def headline(jobs: list[dict], micro: dict) -> dict[str, float | None]:
    """The METRICS values for this run (None where a job is missing)."""
    by_name = {j["name"]: j["cycles"] for j in jobs}
    values: dict[str, float | None] = {}
    for column, (kind, key) in METRICS.items():
        if kind == "job":
            values[column] = by_name.get(key)
        elif kind == "micro":
            values[column] = micro.get(key)
        else:
            stream = [j["cycles"] for j in jobs if j.get("stream") == key]
            values[column] = sum(stream) if stream else None
    return values


def write_results(jobs: list[dict], micro: dict) -> None:
    """Write out/results.json and print a summary."""
    sim = HARDWARE["sim"]
    results = {
        "sim": {"model": sim["value"], "coralnpu_commit": sim["commit"]},
        "microbench": micro,
        "kernels": jobs,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"  {'job':<34}{'cycles':>10}{'err LSB':>9}  ok")
    for j in jobs:
        err = j.get("error_lsb")
        print(
            f"  {j['name']:<34}{j['cycles']:>10}"
            f"{'' if err is None else f'{err:.2f}':>9}  {j.get('ok', '')}"
        )
    print("\n== microbench:", json.dumps(micro))


def cell(value: float | None) -> str:
    """One metric as a log table cell."""
    if value is None:
        return ""
    return f"{value:g}" if isinstance(value, float) else str(value)


def log_row(values: dict[str, float | None], note: str) -> None:
    """Append a row to the RESULTS.md log, with the change vs the last row."""
    lines = LOG.read_text().rstrip("\n").split("\n")

    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    # the log table is the last thing in RESULTS.md: header, rule, rows
    start = max(i for i, line in enumerate(lines) if line.startswith("| date"))
    header = cells(lines[start])
    rows = lines[start + 2 :]
    previous = dict(zip(header, cells(rows[-1]), strict=False)) if rows else {}
    changes = []
    for column, value in values.items():
        old = previous.get(column, "")
        if value is None or not old:
            continue
        delta = (value - float(old)) / float(old)
        if abs(delta) >= CHANGE_THRESHOLD:
            changes.append(f"{column} {delta:+.0%}")

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=SIMS
        ).stdout.strip()

    # "+" marks a run of uncommitted kernel changes on top of that commit
    dirty = git("status", "--porcelain", "--", ".", ":!analysis/RESULTS.md")
    commit = git("rev-parse", "--short", "HEAD") + ("+" if dirty else "")
    row = {
        "date": datetime.date.today().isoformat(),
        "commit": commit,
        "change": note,
        **{k: cell(v) for k, v in values.items()},
        "vs previous": ", ".join(changes) or "no change",
    }
    missing = [c for c in header if c not in row]
    if missing:
        raise SystemExit(f"RESULTS.md log columns not in METRICS: {missing}")
    lines.append("| " + " | ".join(row[c] for c in header) + " |")
    LOG.write_text("\n".join(lines) + "\n")
    print(f"\nlogged to {LOG.name}: {row['vs previous']}")


def main() -> int:
    """Collect everything; exit non-zero if any call was out of tolerance."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", metavar="NOTE", help="append to RESULTS.md")
    args = parser.parse_args()
    jobs = load_jobs()
    if not jobs:
        raise SystemExit(f"no sim outputs in {OUT}: run sim.py test first")
    micro = microbench(jobs)
    write_results(jobs, micro)
    if args.log:
        log_row(headline(jobs, micro), args.log)
    return 0 if all(j.get("ok", True) for j in jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
