`include "reef_defs.svh"

// M3's ClockGate on the sky130 integrated clock gate sky130_fd_sc_hd__sdlclkp_4.
// It replaces ext/coralnpu-sv/ClockGate.sv in sky130 builds (filelist swap, see
// filelists/reef_top_sky130_sram22.f); the upstream file is not edited. Same module name and ports.
//
// sdlclkp: GCLK = CLK & latch(GATE | SCE), latch transparent while CLK is low (.lib
// clock_gating_integrated_cell "latch_posedge_precontrol"). That is the vendored generic branch
// (en_latch = enable | te while !clk_i; clk_o = en_latch & clk_i), with te on the scan enable.
// Drive strength 4 (the largest sdlclkp); CTS buffers the gated clock after it.
// verilator lint_off DECLFILENAME
// verilog_lint: waive module-filename  // must be named ClockGate to replace the vendored one
module ClockGate (
    input  clk_i,
    input  enable,  // '1' passthrough, '0' disable.
    input  te,      // test enable
    output clk_o
);
  sky130_fd_sc_hd__sdlclkp_4 u_cg (
      .CLK (clk_i),
      .GATE(enable),
      .SCE (te),
      .GCLK(clk_o)
  );
endmodule
// verilator lint_on DECLFILENAME
