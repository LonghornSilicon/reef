#!/usr/bin/env bash
# Checks C++ formatting, warnings and clang-tidy without modifying files.
# Run inside the Docker image (tools/docker.sh shell).
set -euo pipefail

cd "$(dirname "$0")/.."

sources=(src/csrc/src/*.cpp tools/*.cpp tests/cpp/*.cpp)
headers=(src/csrc/include/reef_perf/*.hpp)

uv run clang-format --dry-run --Werror "${sources[@]}" "${headers[@]}"

cmake -S . -B build/lint -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_CXX_FLAGS='-Wall -Wextra -Wpedantic -Werror'
cmake --build build/lint

uv run clang-tidy --quiet -p build/lint "${sources[@]}"
