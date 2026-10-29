#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Host C++: the reference engine, its tests and CLI, and coral/'s host tools.
sources=(reference/cpp/*.cpp reference/cpp/tests/*.cpp coral/host/*.cpp)
headers=(reference/cpp/include/inference_engine/*.hpp reference/cpp/include/inference_engine/operator/*.hpp coral/host/*.hpp)
# Coral target code and coral_sim: formatted like the rest, but compiled with
# -Werror by coral/tools/build.sh and setup_sim.sh rather than checked by
# clang-tidy (they need the RISC-V toolchain or Spike's headers).
coral=(coral/include/reef_kernels/*.hpp coral/kernels/* coral/runtime/*.cc coral/firmware/* coral/tests/*.cc coral/tests/*.hpp coral/bench/* coral/sim/*.cpp)

uv run clang-format --dry-run --Werror "${sources[@]}" "${headers[@]}" "${coral[@]}"

cmake -S . -B build/lint -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_CXX_FLAGS='-Wall -Wextra -Wpedantic -Werror'
cmake --build build/lint

tidy_args=()
if [[ "$(uname -s)" == "Darwin" ]]; then
    tidy_args+=(--extra-arg="-isysroot$(xcrun --show-sdk-path)")
    tidy_args+=(--extra-arg=-stdlib=libc++)
fi
uv run clang-tidy --quiet -p build/lint "${tidy_args[@]}" "${sources[@]}"
