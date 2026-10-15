# toy_counter. Tool-neutral SDC, no cell or library names.
create_clock -name clk -period 20.0 [get_ports clk_i]
set_input_delay  -clock clk 4.0 [get_ports {rst_ni en_i}]
set_output_delay -clock clk 4.0 [get_ports count_o*]
