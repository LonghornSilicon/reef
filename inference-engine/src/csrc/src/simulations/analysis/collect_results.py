"""Collect the sim test outputs into results/results.{json,csv}.

    uv run --group sim python analysis/collect_results.py   (from simulations/)

Reads every call record run_tests.py left in out/<test>/*.npz (cycles,
host write cost and the test's own error check), cross-checks flash,
chunked and VME attention against serial on identical inputs, derives the
microbenchmark numbers (load-use latency, VME cycles per matmul_step,
host->DTCM rate), times the host embedding lookup, and writes
analysis/results/results.{json,csv}, which roofline.py reads.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIMS = HERE.parent
OUT = SIMS / "out"
RESULTS = HERE / "results"
HARDWARE = json.loads((HERE / "config" / "hardware.json").read_text())


def embedding_timing() -> list[dict]:
    """Host embedding lookup timing (median/p95 per sequence length)."""
    exe = OUT / "build" / "embedding_bench"
    exe.parent.mkdir(parents=True, exist_ok=True)
    include = SIMS.parent.parent / "include"
    build = [
        "clang++",
        "-std=c++17",
        "-O2",
        f"-I{include}",
        f"-I{SIMS / 'includes'}",
        str(HERE / "host" / "embedding_bench.cc"),
        "-o",
        str(exe),
    ]
    subprocess.run(build, check=True)
    rows = json.loads(subprocess.run([exe], capture_output=True).stdout)
    host = subprocess.run(["uname", "-sm"], capture_output=True, text=True)
    for row in rows:
        row.pop("sink")
        row["host"] = host.stdout.strip()
    return rows


def load_jobs() -> list[dict]:
    """Load every saved call record, with flash/VME vs serial added."""
    jobs, by_name = [], {}
    for path in sorted(OUT.glob("*/*.npz")):
        data = dict(np.load(path))
        job = json.loads(str(data.pop("meta")))
        job["out_bytes"] = sum(
            v.nbytes for k, v in data.items() if k.startswith("out_")
        )
        by_name[job["name"]] = (job, data)
        jobs.append(job)
    # flash and chunked flash vs serial, on identical inputs
    serial = by_name.get("attention_serial_prefill")
    for name in (
        "attention_flash_prefill",
        "chunk_vs_serial_finish",
        "vme_vs_serial_finish",
    ):
        if serial and name in by_name:
            job, data = by_name[name]
            diff = np.abs(data["out_out"].astype(int) - serial[1]["out_out"])
            job["vs_serial_lsb"] = int(diff.max())
            # one rounding step, plus one for VME's per-block uint8 P
            limit = 2 if name.startswith("vme") else 1
            job["ok"] = job.get("ok", True) and job["vs_serial_lsb"] <= limit
    return jobs


def microbench(jobs: list[dict]) -> dict:
    """Derive load-use latency, VME step cost and host->DTCM rate."""
    by = {j["name"]: j for j in jobs}
    out = {}
    for kind in ("scalar", "vector"):
        a, b = by.get(f"load_chain_{kind}_64"), by.get(f"load_chain_{kind}_256")
        if a and b:
            per = (b["cycles"] - a["cycles"]) / (b["loads"] - a["loads"])
            out[f"load_use_{kind}_cycles"] = round(per, 2)
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
    axi = OUT / "ubench" / "axi_write_sweep.json"
    if axi.exists():
        cycles = {
            int(k): v for k, v in json.loads(axi.read_text())["cycles"].items()
        }
        sizes = sorted(cycles)
        slope = np.polyfit(sizes, [cycles[s] for s in sizes], 1)[0]
        out["host_to_dtcm_bytes_per_cycle"] = round(1 / float(slope), 3)
        out["host_to_dtcm_sweep"] = cycles
    return out


def write_results(embed: list[dict], jobs: list[dict], micro: dict) -> None:
    """Write results.json and results.csv and print a summary."""
    sim = HARDWARE["sim"]
    results = {
        "sim": {"model": sim["value"], "coralnpu_commit": sim["commit"]},
        "embedding_host": embed,
        "microbench": micro,
        "kernels": jobs,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    fields = [
        "kernel",
        "name",
        "shape",
        "phase",
        "cycles",
        "ops",
        "host_write_bytes",
        "host_write_cycles",
        "out_bytes",
        "error_lsb",
        "ok",
    ]
    with open(RESULTS / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(jobs)

    print("== embedding (host, p95 ms)")
    for row in embed:
        print(
            f"  {row['phase']:<8} {row['tokens']:>5} tok  "
            f"vocab {row['vocab']:>6}  "
            f"median {row['median_ms']:.4f}  p95 {row['p95_ms']:.4f}"
        )
    print(f"\n  {'job':<34}{'cycles':>10}{'err LSB':>9}  ok")
    for j in jobs:
        err = j.get("error_lsb")
        print(
            f"  {j['name']:<34}{j['cycles']:>10}"
            f"{'' if err is None else f'{err:.2f}':>9}  {j.get('ok', '')}"
        )
    print("\n== microbench:", json.dumps(micro))


def main() -> int:
    """Collect everything; exit non-zero if any call was out of tolerance."""
    jobs = load_jobs()
    if not jobs:
        raise SystemExit(f"no sim outputs in {OUT}: run run_tests.py first")
    write_results(embedding_timing(), jobs, microbench(jobs))
    return 0 if all(j.get("ok", True) for j in jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
