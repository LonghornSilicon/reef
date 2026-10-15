`include "reef_defs.svh"

// toy_counter: WIDTH-bit up counter. Smoke test for lint, sim, synth and the
// PD handoff. Not part of the chip.
// Parameters: WIDTH, default 8.
// Clocking: clk_i rising edge. Reset: rst_ni active low, synchronous.
// Owner: A14N77
module toy_counter #(
    parameter int unsigned WIDTH = 8
) (
    input  logic             clk_i,
    input  logic             rst_ni,
    input  logic             en_i,
    output logic [WIDTH-1:0] count_o
);

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      count_o <= '0;
    end else if (en_i) begin
      count_o <= count_o + WIDTH'(1);
    end
  end

endmodule
