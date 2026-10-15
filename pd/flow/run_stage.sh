#!/usr/bin/env bash
# flow/run_stage.sh <stage> <run_dir>: previous DONE.json must be ok; make
# logs/ reports/ results/; run the stage script; write DONE.json from its exit
# code and results/gate.json.
set -euo pipefail
cd "$(dirname "$0")/.."

stage="$1"; run_dir="$2"
stage_dir="flow/stages/${stage}"
out="${run_dir}/${stage}"
cfg="${run_dir}/run_config.json"

prev="$(find "${run_dir}" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*_*' | sort \
        | awk -v s="${run_dir}/${stage}" '$0 < s' | tail -1 || true)"
if [[ -n "$prev" ]]; then
    status="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" "${prev}/results/DONE.json" 2>/dev/null || echo missing)"
    if [[ "$status" != ok ]]; then
        echo "${stage}: previous stage $(basename "$prev") is '${status}', not ok" >&2
        exit 1
    fi
fi

rm -rf "$out"; mkdir -p "$out"/{logs,reports,results}
run_abs="$(cd "$run_dir" && pwd)"
export REEF_RUN_DIR="$run_abs" REEF_STAGE="$stage" REEF_STAGE_DIR="$run_abs/$stage" \
       REEF_PREV_RESULTS="${prev:+$run_abs/$(basename "$prev")/results}" \
       REEF_CONFIG="$run_abs/run_config.json" REEF_CONFIG_TCL="$run_abs/run_config.tcl"

tool_var="TOOL_${stage}"
start=$(date +%s)
set +e
if [[ -f "${stage_dir}/run.tcl" ]]; then
    ${!tool_var:?no tool for ${stage}} "${stage_dir}/run.tcl" > "${out}/logs/${stage}.log" 2>&1
elif [[ -f "${stage_dir}/run.py" ]]; then
    uv run python "${stage_dir}/run.py" > "${out}/logs/${stage}.log" 2>&1
else
    bash "${stage_dir}/run.sh" > "${out}/logs/${stage}.log" 2>&1
fi
rc=$?
set -e
wall=$(( $(date +%s) - start ))

uv run python flow/done.py --stage "$stage" --out "$out" --config "$cfg" --rc "$rc" --wall "$wall"
status="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" "${out}/results/DONE.json")"
echo "${stage}: ${status} (${wall}s)  ${out}/logs/${stage}.log"
[[ "$status" == ok ]] || { tail -20 "${out}/logs/${stage}.log"; exit 1; }
