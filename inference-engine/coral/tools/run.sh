#!/usr/bin/env bash
# Runs Coral ELFs on reef_perf with the M3 highmem memory map.
#
#   coral/tools/run.sh ELF              reef_perf's full summary
#   coral/tools/run.sh ELF ELF...       one PASS/FAIL line per ELF
#
# A program passes when it reaches mpause. A trap, such as the ebreak of a
# failed golden check, fails it. Exits non-zero if any ELF failed. ELF paths
# are relative to inference-engine/ or absolute inside it.
#
# Runs on $CORAL_SIM if set (coral_sim from setup_sim.sh: pass/fail and an
# instruction count, no timing, as CI uses), else reef_perf from $REEF_PERF
# or PATH, else the reef-perf Docker image that `tools/docker.sh build`
# (repo root) creates.
set -euo pipefail

engine="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$engine"

reef_perf="${CORAL_SIM:-${REEF_PERF:-reef_perf}}"
if ! command -v "$reef_perf" >/dev/null; then
    image="${REEF_PERF_IMAGE:-$(whoami | LC_ALL=C tr -cd 'a-zA-Z0-9')-reef}"
    relative=()
    for elf in "$@"; do
        relative+=("${elf#"$engine"/}")
    done
    exec docker run --rm -v "$engine":/engine -w /engine \
        -e REEF_PERF=/app/reef/reef-perf/build/bin/reef_perf \
        "$image" bash coral/tools/run.sh "${relative[@]}"
fi

# reef_perf writes error-dump.dbg into its working directory when a program
# traps; keep those in the build tree.
mkdir -p build/coral
cd build/coral
if [[ $# -eq 1 ]]; then
    exec "$reef_perf" --highmem --elf "$engine/${1#"$engine"/}"
fi

failed=0
for elf in "$@"; do
    if log="$("$reef_perf" --highmem --elf "$engine/${elf#"$engine"/}" 2>&1)"; then
        retired="$(grep -m1 'instructions retired' <<<"$log" | awk '{print $NF}')"
        echo "PASS $elf ($retired instructions)"
    else
        echo "FAIL $elf: $(grep -m1 -i 'trap\|error' <<<"$log" || tail -1 <<<"$log")"
        failed=1
    fi
done
exit "$failed"
