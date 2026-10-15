#!/usr/bin/env bash
# Die and core from utilization and aspect ratio, IO pins per io_place, macros per macros, taps, power grid.
# In:   10_synth results: netlist, design.sdc
# Out:  results/floorplan.def or .odb
# Gate: every macro legally placed, core non-empty
# Stub.
echo "20_floorplan: not implemented"
echo '{"implemented": false}' > "$REEF_STAGE_DIR/results/gate.json"
