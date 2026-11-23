#!/bin/bash
# run.sh <tag> <elf name in /tmp/study/elfs> [extra -p args]
# Frozen study setup: M3 core, 32-PE Coral array, flash 100 MB/s, PSRAM 50 MB/s.
SIM=${SIM:-/tmp/snap3}; C=$SIM/configs; tag=$1; elf=$2; shift 2
cd /tmp/study/runs
$SIM/reef_perf --elf /tmp/study/elfs/$elf.elf \
  -c $C/machines/m3.yaml -c $C/machines/m3_vme.yaml -c $C/matrix/coral_vme.yaml \
  -c $C/memory/xspi_str.yaml -c $C/memory/psram_kv.yaml -c $C/memory/psram_kv_e2e.yaml \
  -p top.memory.params.host_boot true -p top.memory.params.host_boot_data true \
  -p top.matrix.params.mtx_pe_rows 8 -p top.matrix.params.mtx_pe_cols 4 \
  -p top.memory.params.ext_mem_bytes_per_cycle 2 -p top.memory.params.ext_ram_bytes_per_cycle 1 \
  "$@" --ext-mem-mb 160 --json $tag.json > $tag.log 2>&1
echo "rc=$?" > $tag.done
