#!/usr/bin/env bash
# Applies automatic fixes (Ruff, clang-format), then runs every check.
# Review its edits before committing. Run inside the Docker image.
set -euo pipefail

cd "$(dirname "$0")/.."

uv run ruff check --fix .
uv run ruff format .
uv run ruff check .

sources=(model/*.cpp model/*/*.cpp tools/*/*.cpp tests/cpp/*/*.cpp)
headers=(model/*.h model/*/*.h)

uv run clang-format -i "${sources[@]}" "${headers[@]}"
bash scripts/check_cpp.sh
