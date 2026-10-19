"""Docker + Bazel plumbing shared by build_elfs.py and run_tests.py.

The sim runs in the coralnpu image `coralnpu-reef` (built from coralnpu's
utils/coralnpu.dockerfile) with the volume `reef-coralnpu` holding a
checkout of coralnpu at COMMIT plus the Bazel cache. ~/Projects/coralnpu
(override with CORALNPU=) is only read, to build the image and clone from.
"""

import os
import shutil
import subprocess
from pathlib import Path

SIMS = Path(__file__).resolve().parent.parent
BIN = SIMS / "bin"
OUT = SIMS / "out"
COMMIT = "382b5c12"  # coralnpu main with a working matrix unit
IMAGE, VOLUME = "coralnpu-reef", "reef-coralnpu"
CORALNPU_SRC = Path(
    os.environ.get("CORALNPU", Path.home() / "Projects/coralnpu")
)

# Bash that puts this folder into coralnpu as the Bazel package
# //tests/cocotb/reef: kernels and headers flat next to sim/ and tests/
# files, the ELFs under bin/. /reef is this folder inside the container.
PACKAGE = (
    "set -euo pipefail; pkg=/work/coralnpu/tests/cocotb/reef; "
    'rm -rf "$pkg" && mkdir -p "$pkg/bin"; '
    "cp /reef/kernels/* /reef/includes/* /reef/sim/BUILD /reef/sim/*.cc "
    '/reef/tests/*.py "$pkg"/; '
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
