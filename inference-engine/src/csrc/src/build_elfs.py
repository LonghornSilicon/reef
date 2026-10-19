"""Build the kernel bench ELFs for Coral's VME RTL model into bin/.

    python build_elfs.py [bench_matmul bench_pointwise bench_vme_attention]

With no names it builds all three (see sim/BUILD BENCHES):

    bench_matmul         VME and RVV matmul, and the matmul() dispatcher
    bench_pointwise      lookup-table softmax and GeLU
    bench_vme_attention  block-flash attention on the matrix engine

Each ELF runs one kernel call per run and halts: write the inputs into the
`arena` symbol (20 KB in DTCM) and the call into `bench_args` (op numbers
and argument order are in sim/bench_main.cc); after the halt, `bench_cycles`
holds the kernel's cycle count and the outputs are in the arena.

Needs Docker and a coralnpu checkout at ~/Projects/coralnpu (override with
CORALNPU=). The build runs in the coralnpu image `coralnpu-reef` (built from
coralnpu's utils/coralnpu.dockerfile) with the volume `reef-coralnpu`
holding a checkout of coralnpu at COMMIT plus the Bazel cache; your
checkout is only read, to build the image and clone from. The first run
builds the image and toolchain, which takes a while.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

SIMS = Path(__file__).resolve().parent
BIN = SIMS / "bin"
COMMIT = "382b5c12"  # coralnpu main with a working matrix unit
IMAGE, VOLUME = "coralnpu-reef", "reef-coralnpu"
CORALNPU_SRC = Path(
    os.environ.get("CORALNPU", Path.home() / "Projects/coralnpu")
)
BAZEL = "bazel --output_user_root=/work/cache/bazel"
ELFS = ["bench_matmul", "bench_pointwise", "bench_vme_attention"]

# Bash that puts this folder into coralnpu as the Bazel package
# //tests/cocotb/reef, everything flat. /reef is this folder in the container.
PACKAGE = (
    "set -euo pipefail; pkg=/work/coralnpu/tests/cocotb/reef; "
    'rm -rf "$pkg" && mkdir -p "$pkg"; '
    'cp /reef/kernels/* /reef/includes/* /reef/sim/BUILD /reef/sim/*.cc "$pkg"/; '
    "cd /work/coralnpu; "
)


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


def main() -> int:
    """Build the requested ELFs and copy them into bin/."""
    names = sys.argv[1:] or ELFS
    unknown = [n for n in names if n not in ELFS]
    if unknown:
        raise SystemExit(f"unknown ELF(s) {unknown}; choose from {ELFS}")
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


if __name__ == "__main__":
    sys.exit(main())
