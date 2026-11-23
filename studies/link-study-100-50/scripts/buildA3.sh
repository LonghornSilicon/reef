#!/bin/bash
# Experiment A (and A + int4) on the final design (fwA3), four workloads + small max batch.
S=/tmp/claude-0/-home-user-reef/b2fff131-021a-52a8-bf28-9d9864a85a34/scratchpad/study
export E2E_LDEXTRA=$S/../coral-gcc/lib/gcc/riscv64-unknown-elf/16.1.0/rv32imf_zicsr_zifencei_zmmul_zbb/ilp32/libgcc.a
ST="-DE2E_STREAM_B -DE2E_STREAM_ROWS=64 -DE2E_STREAM_KV -msave-restore -DE2E_LOWRANK"; SMALL="-DE2E_HIDDEN=256 -DE2E_LAYERS=6 -DE2E_VOCAB=4096"
FW=$S/fwA3 EXTRA="$ST -DE2E_KV_BASE=0x24A73800" bash $S/build.sh A3i8 &
FW=$S/fwA3 EXTRA="$ST -DE2E_W4 -DE2E_KV_BASE=0x24A73800" bash $S/build.sh A3i4 &
FW=$S/fwA3 EXTRA="$ST $SMALL -DE2E_KV_BASE=0x20713200" bash $S/build.sh A3si8 &
FW=$S/fwA3 EXTRA="$ST -DE2E_W4 $SMALL -DE2E_KV_BASE=0x20713200" bash $S/build.sh A3si4 &
wait
for v in si8 si4; do X="$ST $SMALL -DE2E_KV_BASE=0x20713200"; [ $v = si4 ] && X="$X -DE2E_W4"
  (cd $S/fwA3/inference-engine && CORAL_TOOLCHAIN=$S/../coral-gcc E2E_OUT=$S/elfs/A3${v}_b15 bash coral/tools/build.sh -DE2E_START_POS=199 -DE2E_NEW_TOKENS=1 -DE2E_STORIES=15 -DE2E_BLOCK_TOKENS=15 -DE2E_MAX_CONTEXT=256 $X > $S/elfs/A3${v}_b15.log 2>&1 && cp $S/elfs/A3${v}_b15/e2e.elf $S/elfs/A3${v}_b15.elf || echo "FAIL A3${v}_b15")
done
ls $S/elfs/A3*.elf | wc -l
