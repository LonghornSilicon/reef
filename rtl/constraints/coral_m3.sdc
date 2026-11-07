# coral_m3: bare CoralNPU M3 RvvCoreMiniAxi. Tool-neutral SDC, no cell or library names.
# Same constraints the Drop 0 DC runs used (rtl/hw-drop0 hw/scripts/synth_dc.sh), at 20 ns.
# One clock, io_aclk. The core gates it internally with ClockGate (RstSync and the
# CSR/WFI gate); tools time those as clock gates, no generated clocks. IO budgets are
# placeholders: the bare core has no pads, reef_top adds them in the follow-up.
create_clock -name clk -period 20.0 [get_ports io_aclk]
set_clock_uncertainty 1.0 [get_clocks clk]
# io_te drives the clock gates' test enable; tied low in functional mode.
set_case_analysis 0 [get_ports io_te]
# io_aresetn: async assert, deassert synchronized by the core's RstSync.
set_false_path -from [get_ports io_aresetn]
set_input_delay  -clock clk 4.0 [remove_from_collection [all_inputs] [get_ports {io_aclk io_aresetn io_te}]]
set_output_delay -clock clk 4.0 [all_outputs]
