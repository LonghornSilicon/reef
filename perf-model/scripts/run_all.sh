#!/usr/bin/env bash
# Runs the perf model on every workload and prints one line per workload.
# Extra arguments are passed to the model, e.g. a config file or overrides:
#
#   scripts/run_all.sh
#   scripts/run_all.sh -c configs/m3.yaml
#   scripts/run_all.sh -p top.core.fetch.params.fetch_interval 2
#
# Per-workload summaries go to results/<workload>.{txt,json}.
set -euo pipefail
source "$(dirname "$0")/env.sh"

RESULTS="${RESULTS:-${PM}/results}"
mkdir -p "${RESULTS}"
cd "${PM}"

printf "%-14s %10s %10s %6s   %s\n" "workload" "insts" "cycles" "IPC" "top dispatch stall"
for elf in workloads/build/*.elf; do
  name="$(basename "${elf}" .elf)"
  json="${RESULTS}/${name}.json"
  if ! "${MODEL}" --elf "${elf}" --json "${json}" "$@" > "${RESULTS}/${name}.txt" 2>&1; then
    echo "${name}: FAILED (see ${RESULTS}/${name}.txt)"
    continue
  fi
  python3 - "${json}" "${name}" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
stalls = {k[len("stall_"):]: v for k, v in d["dispatch"].items() if k.startswith("stall_")}
top = max(stalls, key=stalls.get)
pct = 100.0 * stalls[top] / d["cycles"] if d["cycles"] else 0.0
print(f"{sys.argv[2]:<14} {d['instructions']:>10} {d['cycles']:>10} {d['ipc']:>6.2f}   {top} ({pct:.0f}%)")
EOF
done
