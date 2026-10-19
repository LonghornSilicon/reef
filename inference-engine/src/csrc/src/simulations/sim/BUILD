# Reef kernel benches on Coral's VME RTL model (VmeCoreMiniAxi, Verilator).
#
# build_elfs.py and run_tests.py copy kernels/, includes/, sim/, tests/ and
# bin/ into coralnpu's tests/cocotb/reef/ (kernels flat) and run:
#   bazel build //tests/cocotb/reef:bench_<family>.elf   (build_elfs.py)
#   bazel test //tests/cocotb/reef:<test>_test           (run_tests.py)
# The tests load the ELFs from bin/, so they run on prebuilt ELFs.
# Not meant to be built from the reef repo itself.

load("@coralnpu_hw//third_party/python:requirements.bzl", "requirement")
load("//rules:coco_tb.bzl", "cocotb_test_suite")
load("//rules:coralnpu_v2.bzl", "coralnpu_v2_binary")

KERNEL_HDRS = glob(["*.h"])

# One ELF per kernel family keeps each inside the 8 KB ITCM.
BENCHES = {
    "bench_matmul": (0, [
        "matmul.cc",
        "matmul_rvv.cc",
        "vme_matmul.cc",
    ]),
    "bench_attention": (1, [
        "attention_serial.cc",
        "attention_flash.cc",
        "attention_flash_chunk.cc",
        "online_softmax.cc",
    ]),
    "bench_pointwise": (2, [
        "online_softmax.cc",
        "gelu.cc",
        "layernorm.cc",
        "residual.cc",
        "softmax_lut.cc",
    ]),
    "bench_ubench": (3, []),
    "bench_vme_attention": (4, [
        "vme_attention.cc",
        "attention_flash_chunk.cc",
        "matmul.cc",
        "matmul_rvv.cc",
        "vme_matmul.cc",
    ]),
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

# test -> bench ELF it loads (tests/test_<test>.py)
TESTS = {
    "matmul": "bench_matmul",
    "attention": "bench_attention",
    "vme_attention": "bench_vme_attention",
    "pointwise": "bench_pointwise",
    "ubench": "bench_ubench",
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
