#!/usr/bin/env bash
# Clock tree from cts buffers, NDR on clock nets, post-CTS timing repair.
# In:   30_place results
# Out:  results/cts.def or .odb
# Gate: setup WNS reported, every sink reached
# Stub.
echo "40_cts: not implemented"
echo '{"implemented": false}' > "$REEF_STAGE_DIR/results/gate.json"
