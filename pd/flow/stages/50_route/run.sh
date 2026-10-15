#!/usr/bin/env bash
# Power and special nets first, NDR nets from special_nets, then signal nets, then fillers.
# In:   40_cts results
# Out:  results/route.def, results/<top>_pnr.v
# Gate: zero router DRC violations
# Stub.
echo "50_route: not implemented"
echo '{"implemented": false}' > "$REEF_STAGE_DIR/results/gate.json"
