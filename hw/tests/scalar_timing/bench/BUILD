# Scalar-timing microbenchmarks (hw/tests/scalar_timing/bench in LonghornSilicon/reef);
# bench/sync.sh copies it to tests/cocotb/scalar_timing/ in the build tree.
load("//rules:coco_tb.bzl", "cocotb_test_suite", "verilator_cocotb_model")
load("//rules:coralnpu_v2.bzl", "coralnpu_v2_binary")
load(
    "//tests/cocotb:build_defs.bzl",
    "VCS_BUILD_ARGS",
    "VCS_DEFINES",
    "VCS_TEST_ARGS",
    "VERILATOR_BUILD_ARGS",
)

package(default_visibility = ["//visibility:public"])

# One entry per benchmark; the cocotb testcase has the same name as the ELF.
BENCHES = [
    "fetch_ipc",
    "dep_chain_alu",
    "dep_chain_sw",
    "dep_chain_lw",
    # Deliverable sprint (checklist section 8). All use st_common.h.
    "b1_branch",
    "b2_mul",
    "b3_div",
    "b4_lsu",
    "b4_lsu_misaligned",
    "b5_fpu",
    "b6_csr",
    "b7_rob",
    "b8_x0",
    "b9_alu_base",
    "b9_alu_zbb",
    "b10_system",
    "trace",
]

# name -> (source file, extra compiler flags). The dep_chain_* ELFs share one
# source and differ only in the consumer instruction under test.
SRCS = {
    "fetch_ipc": ("fetch_ipc.cc", []),
    "dep_chain_alu": ("dep_chain.cc", ["-DCONSUMER_ALU"]),
    "dep_chain_sw": ("dep_chain.cc", ["-DCONSUMER_SW"]),
    "dep_chain_lw": ("dep_chain.cc", ["-DCONSUMER_LW"]),
    "b1_branch": ("b1_branch.cc", []),
    "b2_mul": ("b2_mul.cc", []),
    "b3_div": ("b3_div.cc", []),
    "b4_lsu": ("b4_lsu.cc", []),
    "b4_lsu_misaligned": ("b4_lsu_misaligned.cc", []),
    "b5_fpu": ("b5_fpu.cc", []),
    "b6_csr": ("b6_csr.cc", []),
    "b7_rob": ("b7_rob.cc", []),
    "b8_x0": ("b8_x0.cc", []),
    "b9_alu_base": ("b9_alu.cc", ["-DGROUP_BASE"]),
    "b9_alu_zbb": ("b9_alu.cc", ["-DGROUP_ZBB"]),
    "b10_system": ("b10_system.cc", []),
    "trace": ("trace.cc", []),
}

[coralnpu_v2_binary(
    name = b,
    srcs = [SRCS[b][0], "st_common.h"],
    copts = SRCS[b][1],
) for b in BENCHES]

# Same RTL as //tests/cocotb:rvv_core_mini_axi_model, but built with a signal
# tracer. Only the *_waves targets below use it, so the normal runs stay fast.
verilator_cocotb_model(
    name = "rvv_core_mini_axi_trace_model",
    cflags = VERILATOR_BUILD_ARGS,
    hdl_toplevel = "RvvCoreMiniAxi",
    trace = True,
    verilog_source = "//hdl/chisel/src/coralnpu:RvvCoreMiniAxi.sv",
    deps = [
        "//hdl/verilog:sram_backdoor",
    ],
)

cocotb_test_suite(
    name = "scalar_timing_cocotb",
    simulators = ["verilator"],
    testcases = BENCHES,
    testcases_vname = "BENCHES",
    tests_kwargs = {
        "waves": False,
        "hdl_toplevel": "RvvCoreMiniAxi",
        "seed": "42",
        "test_module": ["scalar_timing_test.py"],
        "deps": [
            "//coralnpu_test_utils:core_mini_axi_sim_interface",
            "//coralnpu_test_utils:sim_test_fixture",
            "@rules_python//python/runfiles",
        ],
        "data": [":" + b for b in BENCHES],
        "size": "medium",
    },
    verilator_model = "//tests/cocotb:rvv_core_mini_axi_model",
)

cocotb_test_suite(
    name = "scalar_timing_waves_cocotb",
    simulators = ["verilator"],
    testcases = BENCHES,
    testcases_vname = "BENCHES",
    tests_kwargs = {
        "waves": True,
        # Not part of `bazel test :all`; run a waves target by name.
        "tags": ["manual"],
        "hdl_toplevel": "RvvCoreMiniAxi",
        "seed": "42",
        "test_module": ["scalar_timing_test.py"],
        "deps": [
            "//coralnpu_test_utils:core_mini_axi_sim_interface",
            "//coralnpu_test_utils:sim_test_fixture",
            "@rules_python//python/runfiles",
        ],
        "data": [":" + b for b in BENCHES],
        "size": "medium",
    },
    verilator_model = ":rvv_core_mini_axi_trace_model",
)
