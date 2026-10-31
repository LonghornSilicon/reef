// Cycle model of the OpenRAM macro sky130_sram_2kbyte_1rw1r_32x512_8 for Verilator (2-state,
// no delays). Simulation only; synthesis links the macro's .lib cell instead.
//
// The OpenRAM model (sky130A/libs.ref/sky130_sram_macros/verilog/) uses blocking assignments
// with # delays in its clock blocks, which race against our flops without --timing. This model
// keeps its cycle behaviour without delays:
//   - inputs registered on the rising edge; write and read happen on the falling edge;
//   - dout0 is the read word from the falling edge of the read cycle to the next falling edge.
//     In any other cycle it is POISON (the OpenRAM model drives X right after the next rising
//     edge), so logic that relies on dout0 being held past the cycle after a read fails here.
// tests/sram_sky130 runs the same checks on this model and on the OpenRAM model (VCS).
// verilator lint_off DECLFILENAME
module sky130_sram_2kbyte_1rw1r_32x512_8 (
    input         clk0,
    input         csb0,
    input         web0,
    input  [ 3:0] wmask0,
    input  [ 8:0] addr0,
    input  [31:0] din0,
    output [31:0] dout0,
    // verilator lint_off UNUSEDSIGNAL
    input         clk1,
    input         csb1,
    input  [ 8:0] addr1,
    // verilator lint_on UNUSEDSIGNAL
    output [31:0] dout1
);
  localparam [31:0] POISON = 32'hDEAD_0BAD;

  reg [31:0] mem[0:511];
  reg        csb0_q, web0_q;
  reg [ 3:0] wmask0_q;
  reg [ 8:0] addr0_q;
  reg [31:0] din0_q;
  reg [31:0] dout0_q;

  always @(posedge clk0) begin
    csb0_q   <= csb0;
    web0_q   <= web0;
    wmask0_q <= wmask0;
    addr0_q  <= addr0;
    din0_q   <= din0;
  end

  integer i;
  always @(negedge clk0) begin
    if (!csb0_q && !web0_q) begin
      for (i = 0; i < 4; i = i + 1) if (wmask0_q[i]) mem[addr0_q][8*i+:8] <= din0_q[8*i+:8];
    end
    dout0_q <= (!csb0_q && web0_q) ? mem[addr0_q] : POISON;
  end

  assign dout0 = dout0_q;
  assign dout1 = POISON;  // read port unused in Reef (csb1 tied high)
endmodule
