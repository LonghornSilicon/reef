# Open backend: the binary that runs each stage script. Stubs ignore it.
export TOOL_10_synth := yosys -c
export TOOL_20_floorplan := openroad -exit
export TOOL_30_place := openroad -exit
export TOOL_40_cts := openroad -exit
export TOOL_50_route := openroad -exit
