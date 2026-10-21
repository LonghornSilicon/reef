"""Build and test the Coral NPU kernels.

    python sim.py unit                 host unit tests (clang++, no Docker)
    python sim.py build [elf ...]      build the bench ELFs into bin/
    python sim.py test [test ...]      run the sim tests on the ELFs in bin/

`unit` emulates the vector unit and matrix engine in C (-DRVV_EMULATE
-DVME_EMULATE), so it checks the kernels' logic and indexing, not cycles.

`build` and `test` run in the coralnpu image `coralnpu-reef` (built from
coralnpu's utils/coralnpu.dockerfile) with the volume `reef-coralnpu`
holding a checkout of coralnpu at COMMIT plus the Bazel cache.
~/Projects/coralnpu (override with CORALNPU=) is only read, to build the
image and clone from. The first run builds the RTL model, which takes hours
under amd64 emulation. Each sim test checks every kernel call against NumPy
and fails if any is out of tolerance; logs and per-call .npz records go to
out/<test>/.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

SIMS = Path(__file__).resolve().parent
BIN = SIMS / "bin"
OUT = SIMS / "out"
COMMIT = "382b5c12"  # coralnpu main with a working matrix unit
IMAGE, VOLUME = "coralnpu-reef", "reef-coralnpu"
CORALNPU_SRC = Path(
    os.environ.get("CORALNPU", Path.home() / "Projects/coralnpu")
)

# sim test -> the ELF it loads (BUILD BENCHES / TESTS)
TESTS = {
    "attention": "bench_attention",
    "pointwise": "bench_pointwise",
}
KERNELS = ["matmul.cc", "attention.cc", "pointwise.cc"]
UNIT_FLAGS = ["-std=c++17", "-O2", "-DVME_EMULATE", "-DRVV_EMULATE"]

# Bash that puts this folder into coralnpu as the Bazel package
# //tests/cocotb/reef, the ELFs under bin/. /reef is this folder inside the
# container.
PACKAGE = (
    "set -euo pipefail; pkg=/work/coralnpu/tests/cocotb/reef; "
    'rm -rf "$pkg" && mkdir -p "$pkg/bin"; '
    'cp /reef/*.cc /reef/*.h /reef/*.py /reef/BUILD "$pkg"/; '
    'rm -f "$pkg/test_kernels.cc" "$pkg/sim.py"; '
    'cp /reef/bin/*.elf "$pkg/bin/" 2>/dev/null || true; '
    "cd /work/coralnpu; "
)
BAZEL = "bazel --output_user_root=/work/cache/bazel"


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a command, echoing it, with its output going to the terminal."""
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, text=True)


def docker(script: str) -> int:
    """Run a bash script in the coralnpu image; return its exit code."""
    mounts = ["-v", f"{VOLUME}:/work", "-v", f"{SIMS}:/reef"]
    platform = ["--platform", "linux/amd64"]
    cmd = ["docker", "run", "--rm", *platform, *mounts, IMAGE]
    return run([*cmd, "bash", "-c", PACKAGE + script]).returncode


def prepare() -> None:
    """Build the image and the pinned checkout if needed; exit on failure."""
    if (
        shutil.which("docker") is None
        or subprocess.run(["docker", "info"], capture_output=True).returncode
    ):
        raise SystemExit("docker is not available or its daemon is not running")
    inspect = ["docker", "image", "inspect", IMAGE]
    if subprocess.run(inspect, capture_output=True).returncode:
        dockerfile = CORALNPU_SRC / "utils" / "coralnpu.dockerfile"
        build = [
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
        if run(build).returncode:
            raise SystemExit("docker build failed")
    script = (
        "set -e; git config --global --add safe.directory '*'; "
        "[ -d /work/coralnpu ] || git clone -q /src /work/coralnpu; "
        f"cd /work/coralnpu; git fetch -q /src; git checkout -q {COMMIT}"
    )
    mounts = ["-v", f"{VOLUME}:/work", "-v", f"{CORALNPU_SRC}:/src:ro"]
    clone = ["docker", "run", "--rm", "--platform", "linux/amd64", *mounts]
    if run([*clone, IMAGE, "bash", "-c", script]).returncode:
        raise SystemExit("coralnpu checkout failed")


def unit() -> int:
    """Compile test_kernels.cc with every kernel and run it."""
    exe = OUT / "build" / "test_kernels"
    exe.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["clang++", *UNIT_FLAGS, "-I.", *KERNELS, "test_kernels.cc"]
    if subprocess.run([*cmd, "-o", exe], cwd=SIMS).returncode:
        return 1
    return subprocess.run([exe]).returncode


def build(names: list[str]) -> int:
    """Build the requested ELFs and copy them into bin/."""
    names = names or list(TESTS.values())
    targets = " ".join(f"//tests/cocotb/reef:{n}.elf" for n in names)
    prepare()
    BIN.mkdir(exist_ok=True)
    script = (
        f"{BAZEL} build --jobs=6 {targets}; "
        f"files=$({BAZEL} cquery --output=files 'set({targets})' 2>/dev/null); "
        "for f in $files; "
        'do cp "$f" /reef/bin/ && chmod 644 "/reef/bin/$(basename "$f")"; '
        "done"
    )
    if docker(script):
        return 1
    missing = [n for n in names if not (BIN / f"{n}.elf").exists()]
    if missing:
        print(f"missing from bin/: {missing}")
        return 1
    print("built:", ", ".join(f"bin/{n}.elf" for n in names))
    return 0


def run_test(name: str) -> bool:
    """Run one sim test and copy its outputs to out/<name>/."""
    logs = f"bazel-testlogs/tests/cocotb/reef/{name}_test"
    script = (
        f"{BAZEL} test --jobs=6 --test_timeout=86400 "
        "--nozip_undeclared_test_outputs --test_output=errors "
        f"//tests/cocotb/reef:{name}_test 2>&1 | tail -20; "
        "rc=${PIPESTATUS[0]}; "
        f"rm -rf /reef/out/{name} && mkdir -p /reef/out/{name}; "
        f"cp -r {logs}/test.outputs/. {logs}/test.log /reef/out/{name}/ "
        "2>/dev/null; exit $rc"
    )
    shutil.rmtree(OUT / name, ignore_errors=True)
    return docker(script) == 0


def test(names: list[str]) -> int:
    """Run the requested sim tests and print a pass/fail summary."""
    names = names or list(TESTS)
    unknown = [n for n in names if n not in TESTS]
    if unknown:
        raise SystemExit(
            f"unknown test(s) {unknown}; choose from {list(TESTS)}"
        )
    missing = [
        TESTS[n] for n in names if not (BIN / f"{TESTS[n]}.elf").exists()
    ]
    if missing:
        raise SystemExit(f"no ELF in bin/ for {missing}: run sim.py build")
    prepare()
    status = {n: run_test(n) for n in names}
    for name, ok in status.items():
        print(f"{name:<10} {'pass' if ok else 'FAIL'}  (out/{name}/test.log)")
    return 0 if all(status.values()) else 1


def main() -> int:
    """Dispatch to unit, build or test."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["unit", "build", "test"])
    parser.add_argument("names", nargs="*", help="ELFs or tests (default all)")
    args = parser.parse_args()
    if args.command == "unit":
        return unit()
    if args.command == "build":
        return build(args.names)
    return test(args.names)


if __name__ == "__main__":
    sys.exit(main())
