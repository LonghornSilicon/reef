// M3's TCM `Sram` (128-bit word, 16-bit byte mask, 1-cycle read latency) built from the open
// sky130 OpenRAM macro sky130_sram_2kbyte_1rw1r_32x512_8 (512 x 32 bit, 8-bit write mask).
// It replaces rtl/vendor/coralnpu/Sram.v in sky130 builds (filelist swap, see rtl/sram/README.md);
// the vendored file is not edited. Same module name, parameters and ports.
//
//   ITCM  NUM_ENTRIES =  512:  1 bank  x 4 macros side by side (word bits 32c+31:32c)
//   DTCM  NUM_ENTRIES = 2048:  4 banks x 4 macros, bank = addr[10:9], 4:1 read mux
//
// Only the macro's RW port is used; the read-only port is held deselected.
//
// Read timing: the macro registers its inputs on the rising edge and drives dout0 after the
// falling edge of the access cycle (.lib: timing_type falling_edge). The OpenRAM model drives
// dout0 to X right after the next rising edge, so dout0 is only valid in the second half of the
// access cycle. M3's generic Sram keeps rdata until the next read, so this wrapper keeps the last
// read word in a register: rdata is the macro output in the cycle after a read, the held copy
// otherwise. Unlike the generic model, a write to the address last read does not change rdata
// (same as the TSMC/GF macros the upstream Sram.v uses).
// verilator lint_off DECLFILENAME
module Sram #(
    parameter NUM_ENTRIES = 128,
    // verilator lint_off UNUSEDPARAM
    parameter GLOBAL_BASE_ADDR = 0  // simulation backdoor only (vendored model)
    // verilator lint_on UNUSEDPARAM
) (
    input                            clock,
    input                            enable,
    input                            write,
    input  [$clog2(NUM_ENTRIES)-1:0] addr,
    input  [                  127:0] wdata,
    input  [                   15:0] wmask,
    output [                  127:0] rdata,
    output                           rvalid
);
  localparam int AW = $clog2(NUM_ENTRIES);
  localparam int MACRO_AW = 9;  // 512 words per macro
  localparam int BANKS = NUM_ENTRIES / 512;
  localparam int BW = (BANKS > 1) ? $clog2(BANKS) : 1;

  if (NUM_ENTRIES < 512 || NUM_ENTRIES % 512 != 0 || (BANKS & (BANKS - 1)) != 0) begin : g_bad_size
    $error("Sram_sky130: NUM_ENTRIES must be 512 times a power of two");
  end

  wire [BW-1:0] bank;
  if (BANKS > 1) begin : g_bank_sel
    assign bank = addr[AW-1:MACRO_AW];
  end else begin : g_one_bank
    assign bank = 1'b0;
  end

  wire [127:0] dout[BANKS];

  for (genvar b = 0; b < BANKS; b++) begin : g_bank
    wire csb = ~(enable & (bank == BW'(b)));
    for (genvar c = 0; c < 4; c++) begin : g_col
      sky130_sram_2kbyte_1rw1r_32x512_8 u_macro (
          .clk0  (clock),
          .csb0  (csb),
          .web0  (~write),
          .wmask0(wmask[4*c+:4]),
          .addr0 (addr[MACRO_AW-1:0]),
          .din0  (wdata[32*c+:32]),
          .dout0 (dout[b][32*c+:32]),
          .clk1  (clock),
          .csb1  (1'b1),
          .addr1 ({MACRO_AW{1'b0}}),
          // verilator lint_off PINCONNECTEMPTY
          .dout1 ()
          // verilator lint_on PINCONNECTEMPTY
      );
    end
  end

  reg          rvalid_q;
  reg          rd_q;  // a read was issued last cycle: dout of bank rbank_q is valid now
  reg [BW-1:0] rbank_q;
  reg [ 127:0] hold_q;

  wire [127:0] dout_sel = dout[rbank_q];

  always @(posedge clock) begin
    rvalid_q <= enable;  // same as the vendored Sram
    rd_q     <= enable & ~write;
    if (enable & ~write) rbank_q <= bank;
    if (rd_q) hold_q <= dout_sel;
  end

  assign rdata  = rd_q ? dout_sel : hold_q;
  assign rvalid = rvalid_q;

endmodule
