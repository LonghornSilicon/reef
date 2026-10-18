#!/usr/bin/env bash
# Checks the C++ code against the LLVM coding style without modifying files:
# tools/check_style.sh (names, header guards, file banners), clang-format,
# a warnings-as-errors build, and clang-tidy (naming and llvm-* checks).
# Run inside the Docker image (tools/docker.sh shell).
set -euo pipefail

cd "$(dirname "$0")/.."

sources=(src/*.cpp src/*/*.cpp tools/*.cpp tests/cpp/*/*.cpp)
headers=(src/*.h src/*/*.h)

bash tools/check_style.sh

uv run clang-format --dry-run --Werror "${sources[@]}" "${headers[@]}"

cmake -S . -B build/lint -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_CXX_FLAGS='-Wall -Wextra -Wpedantic -Werror'
cmake --build build/lint

uv run clang-tidy --quiet -p build/lint "${sources[@]}"
