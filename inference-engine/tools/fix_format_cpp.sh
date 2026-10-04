#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

sources=(src/csrc/src/*.cpp src/csrc/src/operator/*.cpp tools/*.cpp tests/cpp/*.cpp)
headers=(src/csrc/include/inference_engine/*.hpp src/csrc/include/inference_engine/operator/*.hpp)

uv run clang-format -i "${sources[@]}" "${headers[@]}"
