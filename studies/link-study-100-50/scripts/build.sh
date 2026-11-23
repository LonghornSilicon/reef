#!/bin/bash
# build.sh <step prefix>: builds b1 b2 b4 b8 ttft32 from study/fw into study/elfs/<prefix>_*.elf and copies them into the container.
S=/tmp/claude-0/-home-user-reef/b2fff131-021a-52a8-bf28-9d9864a85a34/scratchpad; p=$1
FW=${FW:-$S/study/fw}; cd $FW/inference-engine
for b in "b1 -DE2E_START_POS=199 -DE2E_NEW_TOKENS=1 -DE2E_STORIES=1" "b2 -DE2E_START_POS=199 -DE2E_NEW_TOKENS=1 -DE2E_STORIES=2" "b4 -DE2E_START_POS=199 -DE2E_NEW_TOKENS=1 -DE2E_STORIES=4" "b8 -DE2E_START_POS=199 -DE2E_NEW_TOKENS=1 -DE2E_STORIES=8" "ttft32 -DE2E_PROMPT=32 -DE2E_NEW_TOKENS=1"; do
  set -- $b; t=$1; shift
  ( CORAL_TOOLCHAIN=$S/coral-gcc E2E_OUT=$S/study/elfs/${p}_$t bash coral/tools/build.sh "$@" -DE2E_MAX_CONTEXT=256 $EXTRA > $S/study/elfs/${p}_$t.log 2>&1 && cp $S/study/elfs/${p}_$t/e2e.elf $S/study/elfs/${p}_$t.elf && docker cp $S/study/elfs/${p}_$t.elf root-container:/tmp/study/elfs/${p}_$t.elf || { echo "BUILD FAIL $t"; grep -m3 " error\|overflow" $S/study/elfs/${p}_$t.log; } ) &
done; wait
cd $FW && git diff > $S/study/${p}.patch
