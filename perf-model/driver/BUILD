# Bazel package that builds the execute-at-fetch driver as a shared library.
#
# scripts/build.sh copies this directory into the ext/coralnpu-mpact submodule
# as //perf_driver, so it can depend on MPACT's own Bazel targets. The copy is
# untracked there and ignored (.gitmodules: ignore = untracked); edit the
# files here, not the copy.

cc_binary(
    name = "libcoralnpu_driver.so",
    srcs = [
        "coralnpu_driver.cc",
        "coralnpu_driver.h",
    ],
    copts = ["-O2"],
    linkshared = True,
    linkstatic = True,
    deps = [
        "//sim:coralnpu_v2_simulator",
        "@com_google_absl//absl/status:statusor",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

# Small command-line tool that prints the instruction stream MPACT produces.
# Useful for checking a workload before running the perf model.
cc_binary(
    name = "cn_trace",
    srcs = [
        "cn_trace.cc",
        "coralnpu_driver.cc",
        "coralnpu_driver.h",
    ],
    copts = ["-O2"],
    deps = [
        "//sim:coralnpu_v2_simulator",
        "@com_google_absl//absl/status:statusor",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)
