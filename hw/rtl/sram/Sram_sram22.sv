// M3's TCM `Sram` (128-bit word, 16-bit byte mask, 1-cycle read latency) built from the SRAM22
// sky130 macro sram22_512x128m4w8 (512 x 128 bit, 8-bit write mask; ucb-substrate/sram22_sky130_macros,
// pinned in rtl/sram/sram22/UPSTREAM.md). It replaces rtl/vendor/coralnpu/Sram.v in sky130 builds
// (filelist swap, see rtl/sram/README.md); the vendored file is not edited. Same module name,
// parameters and ports.
//
//   ITCM  NUM_ENTRIES =  512:  1 macro
//   DTCM  NUM_ENTRIES = 2048:  4 banks of 1 macro, bank = addr[10:9], 4:1 read mux
//
// The macro matches the interface one to one: active-high ce/we, wmask[i] enables byte i, inputs
// captured on the rising edge, dout launched from the rising edge and held by an output latch until
// the next read. So no hold register is needed (unlike Sram_sky130.sv): rdata is the dout of the
// bank read last. As with the TSMC/GF macros in the vendored Sram.v, a write to the address last
// read does not change rdata.
//
// rstb is tied high. The macro's reset only clears its input flops and control latches (not the
// array or dout), and the generated core gives Sram no reset. See "rstb" in rtl/sram/README.md.
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
    $error("Sram_sram22: NUM_ENTRIES must be 512 times a power of two");
  end

  wire [BW-1:0] bank;
  if (BANKS > 1) begin : g_bank_sel
    assign bank = addr[AW-1:MACRO_AW];
  end else begin : g_one_bank
    assign bank = 1'b0;
  end

  wire [127:0] dout[BANKS];

  for (genvar b = 0; b < BANKS; b++) begin : g_bank
    sram22_512x128m4w8 u_macro (
        .clk  (clock),
        .rstb (1'b1),
        .ce   (enable & (bank == BW'(b))),
        .we   (write),
        .wmask(wmask),
        .addr (addr[MACRO_AW-1:0]),
        .din  (wdata),
        .dout (dout[b])
    );
  end

  reg          rvalid_q;
  reg [BW-1:0] rbank_q;  // bank of the last read: its dout holds the word

  always @(posedge clock) begin
    rvalid_q <= enable;  // same as the vendored Sram
    if (enable & ~write) rbank_q <= bank;
  end

  if (BANKS > 1) begin : g_rmux
    assign rdata = dout[rbank_q];
  end else begin : g_rdirect
    assign rdata = dout[0];
    // verilator lint_off UNUSEDSIGNAL
    wire unused_rbank = ^rbank_q;
    // verilator lint_on UNUSEDSIGNAL
  end
  assign rvalid = rvalid_q;

endmodule
