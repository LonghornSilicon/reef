#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# The engine may be header-only, so these globs may match nothing.
shopt -s nullglob
sources=(src/csrc/src/*.cpp src/csrc/src/operator/*.cpp tools/*.cpp tests/cpp/*.cpp)
headers=(src/csrc/include/inference_engine/*.hpp src/csrc/include/inference_engine/operator/*.hpp)

uv run clang-format --dry-run --Werror "${sources[@]}" "${headers[@]}"

cmake -S . -B build/lint -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_CXX_FLAGS='-Wall -Wextra -Wpedantic -Werror'
cmake --build build/lint

tidy_args=()
if [[ "$(uname -s)" == "Darwin" ]]; then
    tidy_args+=(--extra-arg="-isysroot$(xcrun --show-sdk-path)")
    tidy_args+=(--extra-arg=-stdlib=libc++)
fi
uv run clang-tidy --quiet -p build/lint "${tidy_args[@]}" "${sources[@]}"
