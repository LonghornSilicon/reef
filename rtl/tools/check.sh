#!/usr/bin/env bash
# Check only, no edits. What CI runs. Run inside the image.
set -euo pipefail
cd "$(dirname "$0")/.."

sv_files=()
while IFS= read -r -d '' f; do sv_files+=("$f"); done < <(
    find src \( -name '*.sv' -o -name '*.svh' \) -print0 | sort -z)

echo "== verible"
verible-verilog-lint --rules_config=.rules.verible_lint "${sv_files[@]}"
for f in "${sv_files[@]}"; do verible-verilog-format --verify "$f"; done

echo "== verilator"
for f in filelists/*.f; do
    target="$(basename "$f" .f)"
    top="$(sed -n 's/^top: *//p' "targets/${target}.yaml")"
    args="$(uv run python tools/target_args.py "$target")"  # defines and models
    # shellcheck disable=SC2086  # one word per define or model
    verilator --lint-only -Wall .rules.verilator.vlt $args -f "$f" --top-module "$top"
    tb="$(find src -name "${top}_tb.sv")"
    if [[ -n "$tb" ]]; then
        # shellcheck disable=SC2086
        verilator --lint-only -Wall --timing .rules.verilator.vlt $args -f "$f" "$tb" \
            --top-module "${top}_tb"
    fi
done

echo "== targets"
uv run python tools/validate_handoff.py targets/*.yaml

echo "== ruff"
uv run ruff format --check tools tests
uv run ruff check tools tests

echo "== pytest"
uv run pytest -q

echo "== toy"
make sim TARGET=toy_counter
make synth-check TARGET=toy_counter
