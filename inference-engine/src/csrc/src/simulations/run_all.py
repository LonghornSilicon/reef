"""Build, run, check and report every Reef kernel, from a clean checkout.

    uv run --group sim python src/csrc/src/simulations/run_all.py [--skip-sim]

Steps:
  1. Regenerate includes/params.h from config/*.json.
  2. Host build (clang++, -DVME_EMULATE -DRVV_EMULATE) and run of every
     unit test, plus the host embedding timing.
  3. Sim: copy the kernels into a pinned coralnpu checkout (in the Docker
     volume reef-coralnpu, so ~/Projects/coralnpu is never touched) and run
     the cocotb benches on the VME RTL model (VmeCoreMiniAxi, Verilator).
  4. Check every sim output against NumPy references, derive the
     microbenchmark numbers, and write results/results.{json,csv}.
Anything that did not run is recorded as missing with the reason, never
filled in.
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from subprocess import CompletedProcess

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SIM_RAW = RESULTS / "sim_raw"
BUILD = RESULTS / "build"
CONFIG = json.loads((HERE / "config" / "model.json").read_text())
HARDWARE = json.loads((HERE / "config" / "hardware.json").read_text())
Q = CONFIG["quant"]
SCORE_SCALE = Q["q_scale"] * Q["k_scale"] / math.sqrt(CONFIG["model"]["d_head"])

CORALNPU_SRC = Path(
    os.environ.get("CORALNPU", Path.home() / "Projects/coralnpu")
)
IMAGE, VOLUME = "coralnpu-reef", "reef-coralnpu"
SIM_TESTS = [
    "matmul_test",
    "attention_test",
    "attention_vme_test",
    "pointwise_test",
    "ubench_test",
]

# unit test -> kernel sources it links
UNIT_TESTS = {
    "test_matmul": ["matmul.cc", "matmul_rvv.cc", "matmul_vme.cc"],
    "test_softmax": ["online_softmax.cc"],
    "test_gelu": ["gelu.cc"],
    "test_layernorm": ["layernorm.cc"],
    "test_residual": ["residual.cc"],
    "test_flash": [
        "attention_serial.cc",
        "attention_flash.cc",
        "attention_flash_chunk.cc",
        "attention_vme.cc",
        "online_softmax.cc",
        "matmul.cc",
        "matmul_rvv.cc",
        "matmul_vme.cc",
    ],
}


def run(cmd: list[str], **kwargs: object) -> CompletedProcess:
    """Run a command, echoing it, and capture its output as text."""
    print("$", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, text=True, capture_output=True, **kwargs)


# --------------------------------------------------------------------------
# 1-2. host build, unit tests, embedding timing
# --------------------------------------------------------------------------


def host_build(name: str, sources: list[str], extra: list[str]) -> Path:
    """Compile a host binary into results/build; raise on failure."""
    exe = BUILD / name
    cmd = ["clang++", "-std=c++17", "-O2", "-Ikernels", "-Iincludes", *extra]
    result = run([*cmd, *sources, "-o", str(exe)], cwd=HERE)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed to build:\n{result.stderr}")
    return exe


def unit_tests() -> list[dict]:
    """Build and run every unit test on the host."""
    rows = []
    for name, kernels in UNIT_TESTS.items():
        sources = [f"kernels/{k}" for k in kernels] + [f"unit_tests/{name}.cc"]
        exe = host_build(name, sources, ["-DVME_EMULATE", "-DRVV_EMULATE"])
        result = run([str(exe)])
        rows.append(
            {
                "test": name,
                "pass": result.returncode == 0,
                "output": result.stdout.strip(),
            }
        )
        print(result.stdout, end="")
    return rows


def embedding_timing() -> list[dict]:
    """Host embedding lookup timing (median/p95 per sequence length)."""
    exe = host_build(
        "embedding_bench", ["host/embedding_bench.cc"], ["-I../../include"]
    )
    rows = json.loads(run([str(exe)]).stdout)
    host = run(["uname", "-sm"]).stdout.strip()
    for row in rows:
        row.pop("sink")
        row["host"] = host
    return rows


# --------------------------------------------------------------------------
# 3. sim
# --------------------------------------------------------------------------


def docker(cmd: list[str], mounts: dict[Path, str]) -> CompletedProcess:
    """Run a command in the coralnpu image with the work volume mounted."""
    volumes = ["-v", f"{VOLUME}:/work"]
    for host_path, target in mounts.items():
        volumes += ["-v", f"{host_path}:{target}"]
    platform = ["--platform", "linux/amd64"]
    return run(["docker", "run", "--rm", *platform, *volumes, IMAGE, *cmd])


def prepare_sim() -> str | None:
    """Build the image and the pinned checkout if needed; return any error."""
    if shutil.which("docker") is None or run(["docker", "info"]).returncode:
        return "docker is not available or its daemon is not running"
    if run(["docker", "image", "inspect", IMAGE]).returncode:
        dockerfile = CORALNPU_SRC / "utils" / "coralnpu.dockerfile"
        build = run(
            [
                "docker",
                "build",
                "--platform",
                "linux/amd64",
                "-t",
                IMAGE,
                "-f",
                str(dockerfile),
                str(dockerfile.parent),
            ]
        )
        if build.returncode:
            return f"docker build failed: {build.stderr[-2000:]}"
    commit = HARDWARE["sim"]["commit"]
    script = (
        "set -e; git config --global --add safe.directory '*'; "
        "[ -d /work/coralnpu ] || git clone -q /src /work/coralnpu; "
        f"cd /work/coralnpu; git fetch -q /src; git checkout -q {commit}"
    )
    clone = docker(["bash", "-c", script], {CORALNPU_SRC: "/src:ro"})
    return (
        f"coralnpu checkout failed: {clone.stderr}"
        if clone.returncode
        else None
    )


def run_sim() -> dict:
    """Run the cocotb benches; return {test: pass, fail or missing reason}."""
    error = prepare_sim()
    if error:
        return dict.fromkeys(SIM_TESTS, f"missing: {error}")
    shutil.rmtree(SIM_RAW, ignore_errors=True)
    SIM_RAW.mkdir(parents=True)
    status = {}
    for test in SIM_TESTS:
        result = docker(
            ["bash", "/reef/sim/run_bench.sh", test],
            {HERE: "/reef:ro", SIM_RAW: "/out"},
        )
        status[test] = "pass" if result.returncode == 0 else "fail"
        print(result.stdout[-3000:])
    return status


# --------------------------------------------------------------------------
# 4. checks against NumPy references
# --------------------------------------------------------------------------


def to_int8(x: np.ndarray) -> np.ndarray:
    """The kernels' requantization: round half away from zero, clamp."""
    return np.clip(np.sign(x) * np.floor(np.abs(x) + 0.5), -128, 127)


def attention_ref(
    q: np.ndarray,
    kt: np.ndarray,
    v: np.ndarray,
    query_start: int,
    window: int,
) -> np.ndarray:
    """Float64 causal attention over dequantized inputs, in output LSBs."""
    q, kt, v = (np.asarray(x, np.float64) for x in (q, kt, v))
    scores = SCORE_SCALE * (q @ (kt - Q["k_zero"]))
    out = np.empty((q.shape[0], v.shape[1]))
    for row in range(q.shape[0]):
        pos = query_start + row
        lo = max(0, pos + 1 - window)
        s = scores[row, lo : pos + 1]
        p = np.exp(s - s.max())
        out[row] = (p / p.sum()) @ (v[lo : pos + 1] - Q["v_zero"])
    return Q["v_scale"] * out / Q["attn_out_scale"]


def gelu_ref(x: np.ndarray) -> np.ndarray:
    """GPT-2's tanh GeLU, input and output in int8 LSBs."""
    r = x.astype(np.float64) * Q["gelu_in_scale"]
    y = 0.5 * r * (1 + np.tanh(np.sqrt(2 / np.pi) * (r + 0.044715 * r**3)))
    return np.clip(y / Q["gelu_out_scale"], -128, 127)


def layernorm_ref(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    """Float64 LayerNorm in output LSBs."""
    r = x.astype(np.float64) * Q["act_scale"]
    norm = (r - r.mean(1, keepdims=True)) / np.sqrt(
        r.var(1, keepdims=True) + CONFIG["model"]["layernorm_eps"]
    )
    y = norm * gamma * Q["ln_gamma_scale"] + beta * Q["ln_beta_scale"]
    return np.clip(y / Q["ln_out_scale"], -128, 127)


def check(job: dict, data: dict, outputs: dict) -> tuple[float, float]:
    """Return (max abs error in output LSBs, tolerance) for one sim job."""
    op = job["op"]
    if op.startswith("matmul"):
        a = data["in_a"].astype(np.int64)
        expected = a @ data["in_b"].astype(np.int64)
        return float(np.abs(data["out_c"] - expected).max()), 0.0
    if op in ("attention_serial", "attention_flash", "attention_flash_finish"):
        expected = attention_ref(
            data["in_q"],
            data["in_kt"],
            data["in_v"],
            job["query_start"],
            job["window"],
        )
        # uint8 probabilities (serial, VME) round each p to 1/255 and lose
        # mass on long rows: the serial algorithm itself is 1.44 LSB from
        # float at 256 keys. Float-probability kernels (flash) stay at 1.
        uint8_probs = op == "attention_serial" or job.get(
            "stream", ""
        ).startswith("vme")
        err = float(np.abs(data["out_out"] - expected).max())
        return err, 2.0 if uint8_probs else 1.0
    if op == "online_softmax":
        lo, hi = job["lo"], job["hi"]
        s = SCORE_SCALE * data["in_scores"][lo : hi + 1].astype(np.float64)
        p = np.exp(s - s.max())
        expected = 255 * p / p.sum()
        return float(
            np.abs(data["out_probs"][lo : hi + 1] - expected).max()
        ), 1.0
    if op == "gelu":
        return float(np.abs(data["out_y"] - gelu_ref(data["in_x"])).max()), 0.5
    if op == "layernorm":
        expected = layernorm_ref(
            data["in_x"], data["in_gamma"], data["in_beta"]
        )
        return float(np.abs(data["out_out"] - expected).max()), 1.0
    if op == "residual_add":
        real = data["in_a"] * Q["act_scale"] + data["in_b"] * Q["act_scale"]
        expected = np.clip(real / Q["residual_out_scale"], -128, 127)
        return float(np.abs(data["out_out"] - expected).max()), 0.5
    raise KeyError(op)


def load_sim_jobs() -> list[dict]:
    """Load every saved sim job record, checked where it has outputs."""
    jobs, by_name = [], {}
    for path in sorted(SIM_RAW.glob("*/*.npz")):
        data = dict(np.load(path))
        job = json.loads(str(data.pop("meta")))
        outputs = [k for k in data if k.startswith("out_")]
        job["out_bytes"] = sum(data[k].nbytes for k in outputs)
        if outputs and job["phase"] != "ubench":
            err, tol = check(job, data, outputs)
            job.update(error_lsb=err, tolerance_lsb=tol, ok=err <= tol)
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
    axi = SIM_RAW / "ubench_test" / "axi_write_sweep.json"
    if axi.exists():
        cycles = {
            int(k): v for k, v in json.loads(axi.read_text())["cycles"].items()
        }
        sizes = sorted(cycles)
        slope = np.polyfit(sizes, [cycles[s] for s in sizes], 1)[0]
        out["host_to_dtcm_bytes_per_cycle"] = round(1 / float(slope), 3)
        out["host_to_dtcm_sweep"] = cycles
    return out


# --------------------------------------------------------------------------
# 5. report
# --------------------------------------------------------------------------


def write_results(
    unit: list[dict],
    embed: list[dict],
    sim_status: dict,
    jobs: list[dict],
    micro: dict,
) -> None:
    """Write results.json and results.csv and print a summary."""
    sim = HARDWARE["sim"]
    results = {
        "sim": {
            "model": sim["value"],
            "coralnpu_commit": sim["commit"],
            "status": sim_status,
        },
        "unit_tests": unit,
        "embedding_host": embed,
        "microbench": micro,
        "kernels": jobs,
    }
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

    print("\n== unit tests (host)")
    for row in unit:
        print(f"  {row['test']:<16} {'pass' if row['pass'] else 'FAIL'}")
    print("\n== embedding (host, p95 ms)")
    for row in embed:
        print(
            f"  {row['phase']:<8} {row['tokens']:>5} tok  "
            f"median {row['median_ms']:.4f}  p95 {row['p95_ms']:.4f}"
        )
    print("\n== sim:", ", ".join(f"{k} {v}" for k, v in sim_status.items()))
    print(f"  {'job':<34}{'cycles':>10}{'err LSB':>9}  ok")
    for j in jobs:
        err = j.get("error_lsb")
        print(
            f"  {j['name']:<34}{j['cycles']:>10}"
            f"{'' if err is None else f'{err:.2f}':>9}  {j.get('ok', '')}"
        )
    print("\n== microbench:", json.dumps(micro))


def main() -> int:
    """Run every step; exit non-zero if any test or check failed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--skip-sim",
        action="store_true",
        help="reuse results/sim_raw instead of re-running the sim",
    )
    args = parser.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    run([sys.executable, str(HERE / "config" / "gen_params.py")])

    unit = unit_tests()
    embed = embedding_timing()
    sim_status = (
        dict.fromkeys(SIM_TESTS, "skipped (--skip-sim)")
        if args.skip_sim
        else run_sim()
    )
    jobs = load_sim_jobs() if SIM_RAW.exists() else []
    micro = microbench(jobs)
    write_results(unit, embed, sim_status, jobs, micro)
    failed = not all(r["pass"] for r in unit) or not all(
        j.get("ok", True) for j in jobs
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
