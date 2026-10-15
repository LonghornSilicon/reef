#!/usr/bin/env bash
# Collect GDS, DEF, netlist, SDC, SPEF per corner and reports into export/ with a manifest of git sha, package id, tool versions, checksums.
# In:   60_signoff results
# Out:  export/
# Gate: every listed file present and checksummed
# Stub.
echo "70_export: not implemented"
echo '{"implemented": false}' > "$REEF_STAGE_DIR/results/gate.json"
