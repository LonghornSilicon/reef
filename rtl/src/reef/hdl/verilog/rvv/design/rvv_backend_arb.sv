`ifndef HDL_VERILOG_RVV_DESIGN_RVV_SVH
`include "rvv_backend.svh"
`endif

module rvv_backend_arb(
  clk,
  rst_n,
  req,
  item,
  grant,
  result_valid,
  result
);

// global signal
    input   logic                       clk;
    input   logic                       rst_n;
// PU to ARB
    input   logic     [`NUM_PU-1:0]     req;
    input   PU2ROB_t  [`NUM_PU-1:0]     item;
    output  logic     [`NUM_PU-1:0]     grant;
// ARB to ROB
    output  logic     [`NUM_SMPORT-1:0] result_valid;
    output  PU2ROB_t  [`NUM_SMPORT-1:0] result;

// ---internal signal definition--------------------------------------
  `ifdef ZVE32F_ON
    logic [1:0][1:0]  req_fmamac;
    logic [1:0][1:0]  grant_fmamac;
    logic [1:0]       req_alu;
    logic [1:0]       grant_alu;

    // port 0 
    assign grant[0]      = req[0];
    assign req_fmamac[0] = grant[0] ? 'b0 : {req[8],req[4]};

    arb_round_robin #(.REQ_NUM(2))
    arb_fmamac0 (.grant(grant_fmamac[0]), .req(req_fmamac[0]), .clk(clk), .rst_n(rst_n));

    assign grant[4]        = grant_fmamac[0][0];
    assign grant[8]        = grant_fmamac[0][1];
    assign result_valid[0] = grant[0] || grant[4] || grant[8];
    always_comb begin
      unique case(1'b1)
        grant[0]:           result[0] = item[0];
        grant_fmamac[0][0]: result[0] = item[4];
        default:            result[0] = item[8];
      endcase
    end

     // port 1
    assign grant[1]      = req[1];
    assign req_fmamac[1] = grant[1] ? 'b0 : {req[9],req[5]};

    arb_round_robin #(.REQ_NUM(2))
    arb_fmamac1 (.grant(grant_fmamac[1]), .req(req_fmamac[1]), .clk(clk), .rst_n(rst_n));

    assign grant[5]        = grant_fmamac[1][0];
    assign grant[9]        = grant_fmamac[1][1];
    assign result_valid[1] = grant[1] || grant[5] || grant[9];
    always_comb begin
      unique case(1'b1)
        grant[1]:           result[1] = item[1];
        grant_fmamac[1][0]: result[1] = item[5];
        default:            result[1] = item[9];
      endcase
    end

    // port 2 and port 3
    assign grant[6] = req[6];
    assign grant[7] = req[7];
    assign req_alu  = grant[6]^grant[7] ? req[3:2] : 'b0;
    
    arb_round_robin #(.REQ_NUM(2))
    arb_alu (.grant(grant_alu), .req(req_alu), .clk(clk), .rst_n(rst_n));

    always_comb begin
      case(grant[7:6])
        2'b11: begin
          result_valid[2] = 1'b1;
          result[2]       = item[6];
          result_valid[3] = 1'b1;
          result[3]       = item[7];
          grant[2]        = 'b0;
          grant[3]        = 'b0;
        end
        2'b01: begin
          result_valid[2] = 1'b1;
          result[2]       = item[6];
          result_valid[3] = |grant_alu;
          result[3]       = grant_alu[0] ? item[2] : item[3];
          grant[2]        = grant_alu[0];
          grant[3]        = grant_alu[1];
        end
        2'b10: begin
          result_valid[2] = |grant_alu;
          result[2]       = grant_alu[0] ? item[2] : item[3];
          result_valid[3] = 1'b1;
          result[3]       = item[7];
          grant[2]        = grant_alu[0];
          grant[3]        = grant_alu[1];
        end
        default: begin
          result_valid[2] = req[2];
          result[2]       = item[2];
          result_valid[3] = req[3];
          result[3]       = item[3];
          grant[2]        = req[2];
          grant[3]        = req[3];
        end
      endcase
    end

  `else 

    logic [1:0] req_mac;
    logic [1:0] grant_mac;
    logic [1:0] req_alu;
    logic [1:0] grant_alu;

    // port0 and port1
    assign grant[0] = req[0];
    assign grant[1] = req[1];
    assign req_mac  = grant[0]^grant[1] ? req[5:4] : 'b0;

    arb_round_robin #(.REQ_NUM(2))
    arb_mac (.grant(grant_mac), .req(req_mac), .clk(clk), .rst_n(rst_n));

    always_comb begin
      case(grant[1:0])
        2'b11: begin
          result_valid[0] = 1'b1;
          result[0]       = item[0];
          result_valid[1] = 1'b1;
          result[1]       = item[1];
          grant[4]        = 'b0;
          grant[5]        = 'b0;
        end
        2'b01: begin
          result_valid[0] = 1'b1;
          result[0]       = item[0];
          result_valid[1] = |grant_mac;
          result[1]       = grant_mac[0] ? item[4] : item[5];
          grant[4]        = grant_mac[0];
          grant[5]        = grant_mac[1];
        end
        2'b00: begin
          result_valid[0] = req[4];
          result[0]       = item[4];
          result_valid[1] = req[5];
          result[1]       = item[5];
          grant[4]        = req[4];
          grant[5]        = req[5];
        end
        default: begin
          result_valid[0] = 'b0;
          result[0]       = item[0];
          result_valid[1] = 'b0;
          result[1]       = item[0];
          grant[4]        = 'b0;
          grant[5]        = 'b0;
        end      
      endcase
    end

    // port2 and port3
    assign grant[6] = req[6];
    assign grant[7] = req[7];
    assign req_alu  = grant[6]^grant[7] ? req[3:2] : 'b0;
    
    arb_round_robin #(.REQ_NUM(2))
    arb_alu (.grant(grant_alu), .req(req_alu), .clk(clk), .rst_n(rst_n));

    always_comb begin
      case(grant[7:6])
        2'b11: begin
          result_valid[2] = 1'b1;
          result[2]       = item[6];
          result_valid[3] = 1'b1;
          result[3]       = item[7];
          grant[2]        = 'b0;
          grant[3]        = 'b0;
        end
        2'b01: begin
          result_valid[2] = 1'b1;
          result[2]       = item[6];
          result_valid[3] = |grant_alu;
          result[3]       = grant_alu[0] ? item[2] : item[3];
          grant[2]        = grant_alu[0];
          grant[3]        = grant_alu[1];
        end
        2'b10: begin
          result_valid[2] = |grant_alu;
          result[2]       = grant_alu[0] ? item[2] : item[3];
          result_valid[3] = 1'b1;
          result[3]       = item[7];
          grant[2]        = grant_alu[0];
          grant[3]        = grant_alu[1];
        end
        default: begin
          result_valid[2] = req[2];
          result[2]       = item[2];
          result_valid[3] = req[3];
          result[3]       = item[3];
          grant[2]        = req[2];
          grant[3]        = req[3];
        end
      endcase
    end
  `endif

endmodule
