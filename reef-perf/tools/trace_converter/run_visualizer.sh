#!/usr/bin/env bash
# Converts a `reef_perf --trace` file to a Kanata log and opens it in Konata.
#
#   tools/trace_converter/run_visualizer.sh TRACE.tsv [START_PC END_PC]
#
# Environment:
#   KONATA   path to konata.sh; without it the script only writes the log
#   BUILD    build directory holding bin/trace_converter (default: build)
set -euo pipefail

if [[ $# -ne 1 && $# -ne 3 ]]; then
    echo "Usage: $0 TRACE.tsv [START_PC END_PC]" >&2
    exit 1
fi

root="$(cd "$(dirname "$0")/../.." && pwd)"
trace="$1"
log="${trace%.*}.log"
shift

"${root}/${BUILD:-build}/bin/trace_converter" "${trace}" "${log}" "$@"

if [[ -n "${KONATA:-}" ]]; then
    bash "${KONATA}" "$(realpath "${log}")"
fi
