# Reef kernel benches on Coral's VME RTL model (VmeCoreMiniAxi, Verilator).
#
# run_all.py copies kernels/, includes/, config/*.json and sim/ flat into
# coralnpu's tests/cocotb/reef/ and runs:
#   bazel test //tests/cocotb/reef:reef_bench_<testcase>
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
        "matmul_vme.cc",
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
    ]),
    "bench_ubench": (3, []),
    "bench_attention_vme": (4, [
        "attention_vme.cc",
        "attention_flash_chunk.cc",
        "matmul.cc",
        "matmul_rvv.cc",
        "matmul_vme.cc",
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

TESTCASES = [
    "matmul_test",
    "attention_test",
    "pointwise_test",
    "ubench_test",
    "attention_vme_test",
]

cocotb_test_suite(
    name = "reef_bench",
    simulators = ["verilator"],
    testcases = TESTCASES,
    testcases_vname = "TESTCASES",
    tests_kwargs = {
        "waves": False,
        "hdl_toplevel": "VmeCoreMiniAxi",
        "seed": "42",
        "test_module": ["kernel_bench.py"],
        "deps": [
            "//coralnpu_test_utils:core_mini_axi_sim_interface",
            "//coralnpu_test_utils/sim_backends:verilator_test_fixture",
            "@bazel_tools//tools/python/runfiles",
        ],
        "data": [":" + name for name in BENCHES] + glob(["*.json"]),
        "size": "enormous",
        "timeout": "eternal",
    },
    verilator_model = "//tests/cocotb:vme_core_mini_axi_model",
)
