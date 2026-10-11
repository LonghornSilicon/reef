// Testbench for the sky130 TCM wrappers (rtl/sram/Sram_sram22.sv, Sram_sky130.sv): ITCM (512 x 128)
// and DTCM (2048 x 128) shapes.
// Built by scripts/run_sram_sky130_tb.sh once per wrapper + macro model: Sram_sram22.sv on SRAM22's
// model of its macro, and Sram_sky130.sv on OpenRAM's model (event-driven, with its # delays) and on
// rtl/sram/sim/*_cycle.v. All must pass.
//
// Contract checked, every cycle, 1 ns before the rising edge (where the core samples):
//   - a read of addr returns the word written there (byte mask applied) one cycle later;
//   - rdata keeps that word until the next read, also across idle cycles and writes
//     (including a write to the address just read);
//   - rvalid is enable delayed by one cycle.
`timescale 1ns / 1ps
module tb_sram_sky130;
  localparam real PERIOD = 20.0;  // 50 MHz, the sky130 target

  bit clock = 0;
  always #(PERIOD / 2) clock = ~clock;

  int errors;

  tcm_checker #(.NUM_ENTRIES(512),  .SEED(1), .PERIOD(PERIOD)) u_itcm (.clock);
  tcm_checker #(.NUM_ENTRIES(2048), .SEED(2), .PERIOD(PERIOD)) u_dtcm (.clock);

  initial begin
    wait (u_itcm.done && u_dtcm.done);
    errors = u_itcm.errors + u_dtcm.errors;
    $display("SRAM_TB: itcm %0d random reads, %0d checks, %0d errors; dtcm %0d random reads, %0d checks, %0d errors",
             u_itcm.n_reads, u_itcm.checks, u_itcm.errors, u_dtcm.n_reads, u_dtcm.checks, u_dtcm.errors);
    if (errors == 0 && u_itcm.n_reads > 1000 && u_dtcm.n_reads > 1000) $display("SRAM_TB: PASS");
    else $display("SRAM_TB: FAIL");
    $finish;
  end

  initial begin
    #(PERIOD * 200000);
    $display("SRAM_TB: FAIL (timeout)");
    $finish;
  end
endmodule

module tcm_checker #(
    parameter int NUM_ENTRIES = 512,
    parameter int SEED = 1,
    parameter int RANDOM_OPS = 6000,
    parameter real PERIOD = 20.0
) (
    input bit clock
);
  localparam int AW = $clog2(NUM_ENTRIES);

  logic          enable = 0, write = 0;
  logic [AW-1:0] addr = '0;
  logic [ 127:0] wdata = '0;
  logic [  15:0] wmask = '0;
  wire  [ 127:0] rdata;
  wire           rvalid;

  Sram #(.NUM_ENTRIES(NUM_ENTRIES)) dut (.*);

  logic [127:0] ref_mem[NUM_ENTRIES];
  logic [127:0] expect_q;
  bit           expect_valid = 0;
  bit           prev_enable = 0;
  int           errors = 0, checks = 0;
  int           n_reads = 0;  // random-phase reads, reported so a run without reads can't pass
  bit           done = 0;

  // One access per cycle. Inputs change 2 ns after the rising edge (core flop clk->q).
  task automatic op(input bit en, input bit wr, input logic [AW-1:0] a, input logic [127:0] d,
                    input logic [15:0] m);
    @(posedge clock);
    #2;
    enable = en; write = wr; addr = a; wdata = d; wmask = m;
  endtask

  // Reference: update on the rising edge that captures the access.
  always @(posedge clock) begin
    prev_enable <= enable;
    if (enable && write) begin
      for (int i = 0; i < 16; i++) if (wmask[i]) ref_mem[addr][8*i+:8] <= wdata[8*i+:8];
    end
    if (enable && !write) begin
      expect_q     <= ref_mem[addr];
      expect_valid <= 1;
    end
  end

  // Check 1 ns before each rising edge.
  always @(posedge clock) begin
    #(PERIOD - 1.0);
    if (expect_valid) begin
      checks++;
      if (rdata !== expect_q) begin
        errors++;
        if (errors <= 10)
          $display("%m @%0t: rdata %h, expected %h", $time, rdata, expect_q);
      end
    end
    if (rvalid !== prev_enable) begin
      errors++;
      if (errors <= 10) $display("%m @%0t: rvalid %b, expected %b", $time, rvalid, prev_enable);
    end
  end

  function automatic logic [127:0] rand128();
    return {$urandom, $urandom, $urandom, $urandom};
  endfunction

  initial begin
    void'($urandom(SEED));
    repeat (3) op(0, 0, '0, '0, '0);

    // Fill every word (all bytes known).
    for (int a = 0; a < NUM_ENTRIES; a++) op(1, 1, AW'(a), rand128(), 16'hFFFF);

    // Directed: back-to-back reads across banks, hold over idle cycles, write to the last
    // read address (rdata must not change), byte-masked write then read back.
    op(1, 0, AW'(0), '0, '0);
    op(1, 0, AW'(NUM_ENTRIES - 1), '0, '0);
    op(1, 0, AW'(NUM_ENTRIES / 2), '0, '0);
    repeat (5) op(0, 0, '0, '0, '0);
    op(1, 1, AW'(NUM_ENTRIES / 2), rand128(), 16'hFFFF);
    op(0, 0, '0, '0, '0);
    op(1, 1, AW'(7), rand128(), 16'hA5C3);
    op(1, 0, AW'(7), '0, '0);
    op(1, 1, AW'(7), rand128(), 16'h0000);
    op(1, 0, AW'(7), '0, '0);
    // Enable with write=1 and a zero mask counts as a write (no read): rdata must hold.
    op(1, 1, AW'(3), rand128(), 16'h0000);

    // Random traffic.
    // (Assign in the loop body: a declaration initializer in a static block runs only once.)
    repeat (RANDOM_OPS) begin
      bit en, wr;
      logic [15:0] m;
      en = ($urandom_range(0, 9) < 8);
      wr = ($urandom_range(0, 9) < 4);
      n_reads += en && !wr;
      case ($urandom_range(0, 3))
        0: m = 16'hFFFF;
        1: m = 16'h0000;
        default: m = 16'($urandom);
      endcase
      op(en, wr, AW'($urandom_range(0, NUM_ENTRIES - 1)), rand128(), m);
    end
    op(0, 0, '0, '0, '0);
    repeat (3) @(posedge clock);
    done = 1;
  end
endmodule
