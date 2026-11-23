#!/bin/bash
# build4.sh <prefix> <firmware tree>: builds the four workloads of a step (b1 b2 b4 b8 ttft32 each, plus
# the small model's max batch), named <prefix>i8_*, <prefix>i4_*, <prefix>si8_*, <prefix>si4_*.
S=/tmp/claude-0/-home-user-reef/b2fff131-021a-52a8-bf28-9d9864a85a34/scratchpad/study; p=$1; FW=$2; MAXB=${MAXB:-15}
ST="-DE2E_STREAM_B -DE2E_STREAM_ROWS=64 -DE2E_STREAM_KV $XFLAGS"; SMALL="-DE2E_HIDDEN=256 -DE2E_LAYERS=6 -DE2E_VOCAB=4096"
FW=$FW EXTRA="$ST" bash $S/build.sh ${p}i8 &
FW=$FW EXTRA="$ST -DE2E_W4" bash $S/build.sh ${p}i4 &
FW=$FW EXTRA="$ST $SMALL" bash $S/build.sh ${p}si8 &
FW=$FW EXTRA="$ST -DE2E_W4 $SMALL" bash $S/build.sh ${p}si4 &
wait
for v in si8 si4; do
  X="$ST $SMALL"; [ $v = si4 ] && X="$X -DE2E_W4"
  (cd $FW/inference-engine && CORAL_TOOLCHAIN=$S/../coral-gcc E2E_OUT=$S/elfs/${p}${v}_b$MAXB bash coral/tools/build.sh -DE2E_START_POS=199 -DE2E_NEW_TOKENS=1 -DE2E_STORIES=$MAXB -DE2E_BLOCK_TOKENS=$MAXB -DE2E_MAX_CONTEXT=256 $X > $S/elfs/${p}${v}_b$MAXB.log 2>&1 && cp $S/elfs/${p}${v}_b$MAXB/e2e.elf $S/elfs/${p}${v}_b$MAXB.elf && docker cp $S/elfs/${p}${v}_b$MAXB.elf root-container:/tmp/study/elfs/ || echo "BUILD FAIL ${p}${v}_b$MAXB: $(grep -m1 -i 'error\|DTCM' $S/elfs/${p}${v}_b$MAXB.log)")
done
ls $S/elfs/${p}{i8,i4,si8,si4}_*.elf 2>/dev/null | wc -l
