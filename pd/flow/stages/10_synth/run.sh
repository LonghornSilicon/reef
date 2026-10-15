#!/usr/bin/env bash
# Logic synthesis to a mapped netlist on the std cell library, honoring dont_use.
# In:   00_import results: filelist.f, rtl/, design.sdc
# Out:  results/<top>.v, reports/area and pre-layout timing summary
# Gate: no unmapped cells, no latches
# Stub.
echo "10_synth: not implemented"
echo '{"implemented": false}' > "$REEF_STAGE_DIR/results/gate.json"
