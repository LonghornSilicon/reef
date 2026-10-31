# reef_top SDC v1 (PD Drop 0), sky130 (sky130_fd_sc_hd). Units: ns, pF.
#
# Clock: 20 ns (50 MHz). Design Compiler at the ss corner (100C, 1.60 V) closes 20 ns; 10 ns holds
# only at tt (results/sta_m3_corners_sky130hd.md). Override with REEF_CLK_PERIOD (ns).
# Placeholders until PD has the pad ring: IO budget, driving cell and load.
#
# Clocking: one clock (clk_i). The core gates it internally with two ClockGate instances (in
# RstSync and the CSR/WFI gate), which the sky130 build maps to sky130_fd_sc_hd__sdlclkp_4
# (rtl/clock/). The tool times them as integrated clock gates from the .lib; no generated clocks.
# SPI SCK is NOT a clock: reef_spi2axi oversamples it in clk_i.

if {[info exists ::env(REEF_CLK_PERIOD)]} {
  set clk_period $::env(REEF_CLK_PERIOD)
} else {
  set clk_period 20.0
}

create_clock -name clk -period $clk_period [get_ports clk_i]
set_clock_uncertainty -setup [expr {0.05 * $clk_period}] [get_clocks clk]
set_clock_uncertainty -hold  0.10 [get_clocks clk]
set_clock_transition 0.15 [get_clocks clk]

# ---- Functional mode: test_en_i is tied low on the board (it drives the ICGs' scan enable SCE).
set_case_analysis 0 [get_ports test_en_i]

# ---- Asynchronous inputs, synchronized on chip (2 flops) -> no timing relation to clk.
# rst_ni: async assert; deassert is synchronized by the core's RstSync and reef_rst_sync.
set_false_path -from [get_ports rst_ni]
# SPI pins: 2-flop synchronizers in reef_spi2axi (f_clk >= 16 * f_sck).
set_false_path -from [get_ports {spi_sck_i spi_csn_i spi_mosi_i}]

# ---- Outputs
# spi_miso_o is a flop output; the host samples it about half an SCK period (>= 8 clk) later.
set_false_path -to [get_ports spi_miso_o]
# Status outputs are quasi-static (polled by the host).
set_false_path -to [get_ports {halted_o fault_o wfi_o}]

# ---- IO electrical placeholders (replace with the pad library's values).
set_input_transition 0.5 [all_inputs]
set_load 0.05 [all_outputs]
