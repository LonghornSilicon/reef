#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

uv run ruff check --fix .
uv run ruff format .
uv run ruff check .

sources=(tools/*.cpp tests/*.cpp coral/host/*.cpp)
headers=(include/inference_engine/*.hpp include/inference_engine/operator/*.hpp coral/host/*.hpp)

uv run clang-format -i "${sources[@]}" "${headers[@]}"
bash tools/check_cpp.sh
