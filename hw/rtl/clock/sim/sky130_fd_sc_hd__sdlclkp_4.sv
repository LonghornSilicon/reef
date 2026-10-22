// Model of the sky130 integrated clock gate sky130_fd_sc_hd__sdlclkp_4 for Verilator. Simulation
// only; synthesis links the .lib cell instead.
//
// The PDK model (sky130A/libs.ref/sky130_fd_sc_hd/verilog/sky130_fd_sc_hd.v) is built from a UDP
// latch, which Verilator does not take. Same function: latch GATE | SCE while CLK is low,
// GCLK = latch & CLK. tests/clock_gate_sky130 checks this model and the PDK model (VCS).
// verilator lint_off DECLFILENAME
module sky130_fd_sc_hd__sdlclkp_4 (
    input  CLK,
    input  GATE,
    input  SCE,
    output GCLK
);
  logic en_latch  /* verilator clock_enable */;
  always_latch begin
    if (!CLK) en_latch = GATE | SCE;
  end
  assign GCLK = en_latch & CLK;
endmodule
// verilator lint_on DECLFILENAME
