"""Build the bench ELFs into bin/, without running the sim.

    python build_elfs.py [bench_matmul bench_vme_attention ...]

With no names it builds all five (see sim/BUILD BENCHES). Needs Docker; the
first run builds the coralnpu image and toolchain, which takes a while.
"""

import sys

from sim.coralnpu import BAZEL, BIN, docker, prepare

ELFS = [
    "bench_matmul",
    "bench_attention",
    "bench_vme_attention",
    "bench_pointwise",
    "bench_ubench",
]


def main() -> int:
    """Build the requested ELFs and copy them into bin/."""
    names = sys.argv[1:] or ELFS
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
