# Reef kernel benches on Coral's VME RTL model (VmeCoreMiniAxi, Verilator).
#
# sim.py copies this folder (flat) and bin/ into coralnpu's
# tests/cocotb/reef/ and runs:
#   bazel build //tests/cocotb/reef:bench_<family>.elf   (sim.py build)
#   bazel test //tests/cocotb/reef:<test>_test           (sim.py test)
# The tests load the ELFs from bin/, so they run on prebuilt ELFs.
# Not meant to be built from the reef repo itself.

load("@coralnpu_hw//third_party/python:requirements.bzl", "requirement")
load("//rules:coco_tb.bzl", "cocotb_test_suite")
load("//rules:coralnpu_v2.bzl", "coralnpu_v2_binary")

KERNEL_HDRS = ["helpers.h", "kernels.h"]

# One ELF per kernel family keeps each inside the 8 KB ITCM.
BENCHES = {
    "bench_linear": (0, ["linear.cc"]),
    "bench_attention": (1, ["linear.cc", "attention.cc"]),
    "bench_pointwise": (2, ["activation.cc", "block.cc"]),
    "bench_mlp": (3, ["linear.cc", "activation.cc", "mlp.cc"]),
}

[
    coralnpu_v2_binary(
        name = name,
        srcs = ["bench_main.cc"] + srcs,
        hdrs = KERNEL_HDRS,
        copts = [
            "-O2",
            "-DBENCH_GROUP=%d" % group,
        ],
        stack_size_bytes = 8192,
    )
    for name, (group, srcs) in BENCHES.items()
]

# test -> bench ELF it loads (test_<test>.py)
TESTS = {
    "linear": "bench_linear",
    "attention": "bench_attention",
    "pointwise": "bench_pointwise",
    "mlp": "bench_mlp",
}

[
    cocotb_test_suite(
        name = test,
        simulators = ["verilator"],
        testcases = ["test"],
        tests_kwargs = {
            "waves": False,
            "hdl_toplevel": "VmeCoreMiniAxi",
            "seed": "42",
            "test_module": ["test_%s.py" % test],
            "deps": [
                "//coralnpu_test_utils:core_mini_axi_sim_interface",
                "//coralnpu_test_utils/sim_backends:verilator_test_fixture",
                "@bazel_tools//tools/python/runfiles",
            ],
            "data": ["bench.py", "bin/%s.elf" % elf],
            "size": "enormous",
            "timeout": "eternal",
        },
        verilator_model = "//tests/cocotb:vme_core_mini_axi_model",
    )
    for test, elf in TESTS.items()
]
