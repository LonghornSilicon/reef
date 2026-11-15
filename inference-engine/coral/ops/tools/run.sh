#!/usr/bin/env bash
# Runs one ELF in reef-perf under three memory configs, in parallel:
#   base  : weight flash XSPI STR 1 B/cycle, 68-cycle latency (xspi_str.yaml)
#           + activation/KV PSRAM 0.25 B/cycle, 30/16-cycle latency,
#             100-byte bursts (psram_kv.yaml).  The upstream pessimistic baseline.
#   fast  : flash XSPI DTR 2 B/cycle (xspi_dtr.yaml) + PSRAM at 1 B/cycle.
#   ideal : both links 1024 B/cycle, latency 1: compute-only (roofline ceiling).
#
#   coral/ops/tools/run.sh <elf> [extra reef_perf args...]
# Writes <elf dir>/<elf name>.<cfg>.json and .log. Run from anywhere.
# REEF_PERF_ROOT: a reef-perf tree with build/bin/reef_perf built for Linux
# (yuvesh-kernel-integration @ b9dc6b6).
set -euo pipefail

: "${REEF_PERF_ROOT:=/private/tmp/claude-501/-Users-joshv-Projects-lhs/db0435fb-013c-4ffc-88e2-30356019d7dc/scratchpad/ykw/reef-perf}"
elf="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; shift
dir="$(dirname "$elf")"; name="$(basename "$elf" .elf)"

/usr/local/bin/docker run --rm -v "$REEF_PERF_ROOT":/rp:ro -v "$dir":/w joshv-reef:latest bash -c '
name=$1; shift
rp() { cfg=$1; shift
  /rp/build/bin/reef_perf --elf /w/$name.elf \
    -c /rp/configs/machines/m3.yaml -c /rp/configs/machines/m3_vme.yaml \
    -c /rp/configs/matrix/coral_vme.yaml "$@" \
    -p top.memory.params.host_boot true --ext-mem-mb 160 \
    --json /w/$name.$cfg.json > /w/$name.$cfg.log 2>&1
  echo "exit $?" >> /w/$name.$cfg.log; }
rp base  -c /rp/configs/memory/xspi_str.yaml -c /rp/configs/memory/psram_kv.yaml "$@" &
rp fast  -c /rp/configs/memory/xspi_dtr.yaml -c /rp/configs/memory/psram_kv.yaml \
         -p top.memory.params.ext_ram_bytes_per_cycle 1 "$@" &
rp ideal -c /rp/configs/memory/xspi_str.yaml -c /rp/configs/memory/psram_kv.yaml \
         -p top.memory.params.ext_mem_bytes_per_cycle 1024 -p top.memory.params.ext_mem_latency 1 \
         -p top.memory.params.ext_ram_bytes_per_cycle 1024 -p top.memory.params.ext_ram_latency 1 \
         -p top.memory.params.ext_ram_write_latency 1 -p top.memory.params.ext_ram_max_burst_bytes 65536 "$@" &
wait' _ "$name" "$@"
for cfg in base fast ideal; do
  grep -q "exit 0" "$dir/$name.$cfg.log" || { echo "FAILED $cfg: $dir/$name.$cfg.log"; tail -5 "$dir/$name.$cfg.log"; }
done
python3 "$(dirname "$0")/stats.py" "$dir/$name".{base,fast,ideal}.json
