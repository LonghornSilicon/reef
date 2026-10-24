`include "reef_defs.svh"

// reef_rst_sync: async assert, sync deassert (2 flops) for the logic reef_top adds around the
// core. The core has its own RstSync on io_aresetn.
// In test mode the raw pin reset is used, so scan can control reset directly.
module reef_rst_sync (
    input  logic clk_i,
    input  logic rst_ni,
    input  logic test_en_i,
    output logic rst_no
);
  logic [1:0] q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) q <= 2'b00;
    else q <= {q[0], 1'b1};
  end
  assign rst_no = test_en_i ? rst_ni : q[1];
endmodule
