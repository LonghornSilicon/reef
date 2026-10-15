#!/usr/bin/env bash
# Check only, no edits. What CI runs. Run inside the image.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== shellcheck"
shellcheck docker/*.sh flow/run_stage.sh flow/stages/*/run.sh tools/*.sh

echo "== yamllint"
yamllint -c .yamllint tech designs

echo "== ruff"
uv run ruff format --check flow tools tests
uv run ruff check flow tools tests

echo "== config"
uv run python flow/config.py --tech tech/sky130 --design designs/toy_counter/design.yaml \
    --run-id check --out /tmp/reef_check_config.json > /dev/null

echo "== pytest"
uv run pytest -q

echo "== private guard"
bash tools/check_no_private.sh
