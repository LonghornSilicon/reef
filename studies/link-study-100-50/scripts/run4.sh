#!/bin/bash
# run4.sh <prefix> [maxb]: runs a step's 22 builds in the container, 4 at a time, on snap10.
S=/tmp/claude-0/-home-user-reef/b2fff131-021a-52a8-bf28-9d9864a85a34/scratchpad/study; p=$1
CA=$(cat $S/core_args.txt); B="-p top.memory.params.dma_channels 2 -p top.matrix.params.mtx_stream_b true"
bash $S/jobs4.sh $p ${2:-15} > $S/jobs_$p.txt; docker cp $S/jobs_$p.txt root-container:/tmp/study/jobs_$p.txt
docker exec -d root-container bash -c "cd /tmp/study && xargs -P 4 -L 1 bash -c 'SIM=\${SIM:-/tmp/snap10} bash run.sh \$0 \$1 $CA $B \${@:2}' < jobs_$p.txt"
