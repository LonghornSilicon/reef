# One clock domain, everything on io_aclk. Period is a placeholder until RTL
# and PD agree. Async reset is not timed. No IO delays: the AXI ports face
# whatever the chip top puts there, and that target carries its own SDC.
create_clock -name aclk -period 20.0 [get_ports io_aclk]
set_false_path -from [get_ports io_aresetn]
