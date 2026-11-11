#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Host C++: the reference engine, its tests and CLI.
sources=(reference/cpp/*.cpp reference/cpp/tests/*.cpp)
headers=(reference/cpp/include/inference_engine/*.hpp reference/cpp/include/inference_engine/operator/*.hpp)
# Coral target code: formatted like the rest, but compiled with -Werror by
# coral/tools/build.sh rather than checked by clang-tidy (it needs the RISC-V
# toolchain).
coral=(coral/kernels/* coral/runtime/*.cc)

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
