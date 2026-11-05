#!/usr/bin/env bash
# Converts a `reef_perf --trace` file to a Kanata log and opens it in Konata.
#
#   tools/trace_converter/run_visualizer.sh TRACE.tsv [START_SEQ END_SEQ]
#
# Environment:
#   KONATA   path to konata.sh; it serves the log on a local port and the
#            script opens the browser there. Without it the script only
#            writes the log.
#   BUILD    build directory holding bin/trace_converter (default: build)
set -euo pipefail

if [[ $# -ne 1 && $# -ne 3 ]]; then
    echo "Usage: $0 TRACE.tsv [START_SEQ END_SEQ]" >&2
    exit 1
fi

root="$(cd "$(dirname "$0")/../.." && pwd)"
trace="$1"
log="${trace%.*}.log"
shift

"${root}/${BUILD:-build}/bin/trace_converter" "${trace}" "${log}" "$@"

if [[ -n "${KONATA:-}" ]]; then
    # konata.sh serves the log on a local port and blocks until Ctrl+C. Pick
    # the port here so the browser can be opened on the right URL.
    export KONATA_PORT="${KONATA_PORT:-$(python3 -c '
import socket
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])')}"
    name="$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1]))' "$(basename "${log}")")"
    url="http://127.0.0.1:${KONATA_PORT}/#name=${name}"
    (
        sleep 1
        if command -v xdg-open >/dev/null; then xdg-open "${url}"
        elif command -v wslview >/dev/null; then wslview "${url}"
        elif command -v explorer.exe >/dev/null; then explorer.exe "${url}"
        fi
    ) >/dev/null 2>&1 &
    exec bash "${KONATA}" "$(realpath "${log}")"
fi
