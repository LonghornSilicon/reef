# Reef kernel bench ELFs for Coral's VME RTL model (VmeCoreMiniAxi).
#
# build_elfs.py copies kernels/, includes/ and sim/ into coralnpu's
# tests/cocotb/reef/ (all flat) and runs:
#   bazel build //tests/cocotb/reef:bench_<family>.elf
# Not meant to be built from the reef repo itself.

load("//rules:coralnpu_v2.bzl", "coralnpu_v2_binary")

# Kernels are header-only; bench_main.cc includes the ones its group needs.
# One ELF per kernel family keeps each inside the 8 KB ITCM.
BENCHES = {
    "bench_matmul": 0,
    "bench_pointwise": 2,
    "bench_vme_attention": 4,
}

[
    coralnpu_v2_binary(
        name = name,
        srcs = ["bench_main.cc"],
        hdrs = glob(["*.h"]),
        copts = [
            "-O2",
            "-DBENCH_GROUP=%d" % group,
        ],
        stack_size_bytes = 8192,
    )
    for name, group in BENCHES.items()
]
