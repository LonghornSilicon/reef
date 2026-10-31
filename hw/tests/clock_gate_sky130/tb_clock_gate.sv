// Testbench for M3's ClockGate contract, run by scripts/run_clock_gate_sky130_tb.sh on:
// the vendored generic ClockGate.sv, rtl/clock/ClockGate_sky130.sv with the PDK's
// sky130_fd_sc_hd__sdlclkp_4 model (functional and timing variants) and with our Verilator model
// in rtl/clock/sim/. All must pass; a mutant cell without the latch must fail.
//
// Contract: clk_o = clk_i & en, where en follows (enable | te) while clk_i is low and holds while
// clk_i is high. So a change of enable or te while clk_i is high must not reach clk_o (no glitch,
// no truncated pulse) until the next low phase.
//
// enable and te change at random times on a 0.1 ns grid offset by 0.03 ns (never on a clock edge),
// in both clock phases. clk_o is compared with the reference every 0.1 ns, offset by 0.07 ns.
`timescale 1ns / 1ps
module tb_clock_gate;
  localparam real PERIOD = 20.0;  // 50 MHz, the sky130 target
  localparam int CYCLES = 20000;

  logic clk_i = 0, enable = 0, te = 0;
  wire  clk_o;

  ClockGate dut (.*);

  always #(PERIOD / 2) clk_i = ~clk_i;

  // Reference latch: transparent while clk_i is low.
  logic ref_en = 0;
  always @(clk_i or enable or te) if (!clk_i) ref_en = enable | te;
  wire ref_clk = ref_en & clk_i;

  int errors = 0, samples = 0;
  int chg_high = 0, chg_low = 0;  // stimulus changes while clk_i is high / low
  int cyc_on = 0, cyc_off = 0, cyc_te_only = 0;  // rising edges by reference gate state
  int pulses_dut = 0, pulses_ref = 0;

  // Stimulus: change enable and/or te at random grid points.
  initial begin
    #0.03;
    forever begin
      #(0.1 * $urandom_range(1, 150));
      case ($urandom_range(0, 2))
        0: enable = ~enable;
        1: te = ~te;
        default: begin
          enable = $urandom_range(0, 1);
          te = ($urandom_range(0, 3) == 0);
        end
      endcase
      if (clk_i) chg_high++;
      else chg_low++;
    end
  end

  // Compare every 0.1 ns.
  initial begin
    #0.07;
    forever begin
      samples++;
      if (clk_o !== ref_clk) begin
        errors++;
        if (errors <= 10)
          $display("CG_TB: mismatch at %0.2f ns: clk_i=%b enable=%b te=%b clk_o=%b expected=%b",
                   $realtime, clk_i, enable, te, clk_o, ref_clk);
      end
      #0.1;
    end
  end

  always @(posedge clk_i) begin
    if (!ref_en) cyc_off++;
    else if (te && !enable) cyc_te_only++;
    else cyc_on++;
  end
  always @(posedge clk_o) pulses_dut++;
  always @(posedge ref_clk) pulses_ref++;

  initial begin
    #(PERIOD * CYCLES);
    $display("CG_TB: %0d cycles, %0d samples, %0d errors; clk_o pulses %0d (expected %0d)",
             CYCLES, samples, errors, pulses_dut, pulses_ref);
    $display("CG_TB: coverage: rising edges gated on %0d / off %0d / on by te only %0d; input changes while clk high %0d, low %0d",
             cyc_on, cyc_off, cyc_te_only, chg_high, chg_low);
    if (errors == 0 && pulses_dut == pulses_ref && cyc_on > 1000 && cyc_off > 1000 && cyc_te_only > 100
        && chg_high > 1000 && chg_low > 1000)
      $display("CG_TB: PASS");
    else $display("CG_TB: FAIL");
    $finish;
  end
endmodule
