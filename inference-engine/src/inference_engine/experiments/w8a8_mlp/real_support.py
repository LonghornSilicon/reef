"""Artifact integrity, host builds and proposed Coral buffer accounting."""

import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

REEF_ROOT = Path(__file__).resolve().parents[5]
SIMULATIONS = REEF_ROOT / "inference-engine/src/csrc/src/simulations"
ATOL = 2e-6
RTOL = 2e-5


def sha256(path: Path) -> str:
    """Hash a file without loading a checkpoint into memory."""
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def external_directory(path: Path) -> Path:
    """Require generated arrays and native artifacts outside the checkout."""
    resolved = path.resolve()
    if resolved.is_relative_to(REEF_ROOT):
        raise ValueError(
            "Generated artifacts must be outside the Reef checkout"
        )
    return resolved


def build_host_library(destination: Path) -> dict[str, object]:
    """Compile the existing scalar kernel with a thin C ABI bridge."""
    destination = external_directory(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which("c++")
    if compiler is None:
        raise RuntimeError("A host C++ compiler is required")
    sources = [
        SIMULATIONS / "kernels/mlp_w8a8.cc",
        SIMULATIONS / "validation/host_bridge.cc",
    ]
    command = [
        compiler,
        "-std=c++17",
        "-O2",
        "-ffp-contract=off",
        "-fPIC",
        "-shared",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
        *(str(source) for source in sources),
        "-lm",
        "-o",
        str(destination),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    version = subprocess.check_output([compiler, "--version"], text=True)
    return {
        "command": command,
        "compiler": version.splitlines()[0],
        "library_sha256": sha256(destination),
        "source_sha256": {
            str(p.relative_to(REEF_ROOT)): sha256(p) for p in sources
        },
    }


def validate_memory_plan(path: Path) -> dict[str, object]:
    """Check every proposed buffer's size, alignment, bounds and overlap."""
    plan = json.loads(path.read_text())
    if plan["dimensions"] != {"tokens": 3, "width": 512, "hidden": 2048}:
        raise ValueError("Memory plan must describe the approved real block")
    sizes = {"fp32": 4, "int32": 4, "int8": 1, "bytes": 1}
    expected = {
        "w1": [2048, 512],
        "s1": [2048],
        "b1": [2048],
        "w2": [512, 2048],
        "s2": [512],
        "b2": [512],
        "a1": [3, 2048],
        "hidden": [3, 2048],
        "hidden_q": [3, 2048],
        "pre": [3, 2048],
        "input_fp32": [3, 512],
        "input_q": [3, 512],
        "input_scale": [3],
        "hidden_scale": [3],
        "a2": [3, 512],
        "output": [3, 512],
        "stack": [2048],
        "runtime_reserve": [10720],
    }
    regions = plan["regions"]
    types = {
        name: (
            "int8"
            if name in {"w1", "w2", "input_q", "hidden_q"}
            else "int32"
            if name in {"a1", "a2"}
            else "bytes"
            if name in {"stack", "runtime_reserve"}
            else "fp32"
        )
        for name in expected
    }
    if [(r["name"], int(r["address"], 0), r["bytes"]) for r in regions] != [
        ("ITCM", 0, 8192),
        ("DTCM", 0x10000, 32768),
        ("EXTMEM", 0x20000000, 4194304),
    ]:
        raise ValueError("Plan must retain the inspected default memory map")
    by_region = {region["name"]: [] for region in regions}
    names = set()
    for buffer in plan["buffers"]:
        name = buffer["name"]
        if (
            name not in expected
            or name in names
            or buffer["shape"] != expected[name]
        ):
            raise ValueError(f"Unexpected or duplicate buffer shape: {name}")
        if (
            buffer["dtype"] != types[name]
            or buffer["alignment"] < sizes[types[name]]
        ):
            raise ValueError(f"Incorrect buffer type/alignment: {name}")
        names.add(name)
        alignment = buffer["alignment"]
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError(f"Invalid alignment: {name}")
        count = math.prod(buffer["shape"]) * sizes[buffer["dtype"]]
        if count != buffer["bytes"]:
            raise ValueError(f"Incorrect byte count: {name}")
        start = int(buffer["address"], 0)
        if start % buffer["alignment"]:
            raise ValueError(f"Misaligned buffer: {name}")
        by_region[buffer["region"]].append((start, start + count, name))
    if names != set(expected):
        raise ValueError("Missing memory-plan buffers")
    usage = {}
    for region in regions:
        start = int(region["address"], 0)
        previous_end = start
        used = 0
        for low, high, name in sorted(by_region[region["name"]]):
            if low < previous_end or high > start + region["bytes"]:
                raise ValueError(f"Overlapping or out-of-range buffer: {name}")
            previous_end = high
            used += high - low
        usage[region["name"]] = {
            "reserved_bytes": used,
            "remaining_bytes": region["bytes"] - used,
        }
    return {
        "status": "PASS",
        "plan_sha256": sha256(path),
        "regions": usage,
        "scope": "static placement only; no guest ELF or simulator run",
    }
