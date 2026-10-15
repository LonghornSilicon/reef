#!/usr/bin/env bash
# Auto-fix formatting (verible, ruff), then run every check. Run inside the image.
set -euo pipefail
cd "$(dirname "$0")/.."

find src \( -name '*.sv' -o -name '*.svh' \) -print0 | xargs -0 verible-verilog-format --inplace
uv run ruff check --fix tools tests
uv run ruff format tools tests

bash tools/check.sh
