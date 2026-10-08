#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

uv run ruff check --fix .
uv run ruff format .
uv run ruff check .

# The engine is header-only, so these globs may match nothing.
shopt -s nullglob
sources=(src/csrc/src/*.cpp src/csrc/src/operator/*.cpp tools/*.cpp tests/cpp/*.cpp)
headers=(src/csrc/include/inference_engine/*.hpp src/csrc/include/inference_engine/operator/*.hpp)

uv run clang-format -i "${sources[@]}" "${headers[@]}"
bash tools/check_cpp.sh
