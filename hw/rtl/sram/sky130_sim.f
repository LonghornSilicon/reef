// sky130 TCM (Verilator simulation): the wrapper plus our cycle model of the OpenRAM macro.
// Event-driven simulators with timing (VCS, Xcelium) can use OpenRAM's own model instead.
${SRAM_RTL}/Sram_sky130.sv
${SRAM_RTL}/sim/sky130_sram_2kbyte_1rw1r_32x512_8_cycle.v
