`include "reef_defs.svh"

// Self-checking test for toy_counter, beside the module it tests. $fatal on
// mismatch, so the sim exits nonzero. Run with `make sim TARGET=toy_counter`.
module toy_counter_tb;
  logic       clk;
  logic       rst_n;
  logic       en;
  logic [7:0] count;

  toy_counter dut (
      .clk_i  (clk),
      .rst_ni (rst_n),
      .en_i   (en),
      .count_o(count)
  );

  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

  task automatic expect_count(input logic [7:0] exp);
    if (count !== exp) $fatal(1, "count=%0d expected %0d", count, exp);
  endtask

  initial begin
    rst_n = 1'b0;
    en = 1'b0;
    repeat (2) @(posedge clk);
    #1 expect_count(8'd0);
    rst_n = 1'b1;
    repeat (3) @(posedge clk);
    #1 expect_count(8'd0);
    en = 1'b1;
    repeat (10) @(posedge clk);
    #1 expect_count(8'd10);
    en = 1'b0;
    repeat (3) @(posedge clk);
    #1 expect_count(8'd10);
    en = 1'b1;
    repeat (246) @(posedge clk);
    #1 expect_count(8'd0);
    rst_n = 1'b0;
    @(posedge clk);
    #1 expect_count(8'd0);
    $display("PASS toy_counter_tb");
    $finish;
  end
endmodule
