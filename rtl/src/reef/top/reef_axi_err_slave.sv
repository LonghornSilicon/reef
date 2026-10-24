`include "reef_defs.svh"

// reef_axi_err_slave: terminates the core's AXI master port on the test chip.
//
// There is no off-chip memory in v1, so every access outside ITCM/DTCM/CSR gets DECERR.
// Bursts are honored (len+1 read beats with last; write data drained up to wlast), one
// transaction per channel at a time. The core turns DECERR into a load/store/fetch fault.
module reef_axi_err_slave #(
    parameter int IdW = 6
) (
    input logic clk_i,
    input logic rst_ni,

    input  logic           aw_valid_i,
    output logic           aw_ready_o,
    input  logic [IdW-1:0] aw_id_i,
    input  logic           w_valid_i,
    output logic           w_ready_o,
    input  logic           w_last_i,
    output logic           b_valid_o,
    input  logic           b_ready_i,
    output logic [IdW-1:0] b_id_o,
    output logic [    1:0] b_resp_o,

    input  logic           ar_valid_i,
    output logic           ar_ready_o,
    input  logic [IdW-1:0] ar_id_i,
    input  logic [    7:0] ar_len_i,
    output logic           r_valid_o,
    input  logic           r_ready_i,
    output logic [IdW-1:0] r_id_o,
    output logic [  127:0] r_data_o,
    output logic [    1:0] r_resp_o,
    output logic           r_last_o
);

  localparam logic [1:0] DecErr = 2'b11;

  // Write: accept AW, drain W until last, then send B.
  typedef enum logic [1:0] {
    WIdle,
    WData,
    WResp
  } w_e;
  w_e wst;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      wst    <= WIdle;
      b_id_o <= '0;
    end else begin
      unique case (wst)
        WIdle:
        if (aw_valid_i) begin
          b_id_o <= aw_id_i;
          wst <= WData;
        end
        WData:   if (w_valid_i && w_last_i) wst <= WResp;
        WResp:   if (b_ready_i) wst <= WIdle;
        default: wst <= WIdle;
      endcase
    end
  end
  assign aw_ready_o = (wst == WIdle);
  assign w_ready_o  = (wst == WData);
  assign b_valid_o  = (wst == WResp);
  assign b_resp_o   = DecErr;

  // Read: accept AR, return len+1 error beats.
  logic       r_busy;
  logic [7:0] r_left;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      r_busy <= 1'b0;
      r_left <= '0;
      r_id_o <= '0;
    end else if (!r_busy) begin
      if (ar_valid_i) begin
        r_busy <= 1'b1;
        r_left <= ar_len_i;
        r_id_o <= ar_id_i;
      end
    end else if (r_ready_i) begin
      if (r_left == 8'd0) r_busy <= 1'b0;
      else r_left <= r_left - 8'd1;
    end
  end
  assign ar_ready_o = ~r_busy;
  assign r_valid_o  = r_busy;
  assign r_data_o   = '0;
  assign r_resp_o   = DecErr;
  assign r_last_o   = (r_left == 8'd0);

endmodule
