#!/usr/bin/env bash
# Multi-corner STA from extracted parasitics, DRC, LVS, antenna.
# In:   50_route results
# Out:  results/<top>.gds, reports/sta_<corner>.rpt, drc.rpt, lvs.rpt, antenna.rpt
# Gate: setup and hold WNS >= 0, DRC clean, LVS clean
# Stub.
echo "60_signoff: not implemented"
echo '{"implemented": false}' > "$REEF_STAGE_DIR/results/gate.json"
