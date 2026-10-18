#!/usr/bin/env bash
# Applies automatic fixes (Ruff, clang-format), then runs every check.
# Review its edits before committing. Run inside the Docker image.
set -euo pipefail

cd "$(dirname "$0")/.."

uv run ruff check --fix .
uv run ruff format .
uv run ruff check .

sources=(src/*.cpp src/*/*.cpp tools/*.cpp tests/cpp/*/*.cpp)
headers=(src/*.h src/*/*.h)

uv run clang-format -i "${sources[@]}" "${headers[@]}"
bash tools/check_cpp.sh
