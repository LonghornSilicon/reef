module ClockGate_0 ( clk_i, enable, clk_o, te_BAR );
  input clk_i, enable, te_BAR;
  output clk_o;
  wire   te, n3, n4;
  assign te = te_BAR;

  sky130_fd_sc_hd__dlxtn_1 en_latch_reg ( .D(n3), .GATE_N(clk_i), .Q(n4) );
  sky130_fd_sc_hd__nor2b_1 U2 ( .B_N(te), .A(enable), .Y(n3) );
  sky130_fd_sc_hd__and2b_2 U3 ( .B(clk_i), .A_N(n4), .X(clk_o) );
endmodule
module RstSync ( clk_i, rstn_i, clk_en, clk_o, rstn_o, te_BAR );
  input clk_i, rstn_i, clk_en, te_BAR;
  output clk_o, rstn_o;
  wire   te, rst_delay_reg_0, n5;
  wire   [3:2] rst_delay_reg;
  assign te = te_BAR;

  ClockGate_0 icg ( .clk_i(clk_i), .enable(rst_delay_reg[3]), .clk_o(clk_o), 
        .te_BAR(te) );
  sky130_fd_sc_hd__dfrtp_1 \rst_delay_reg_reg[0]  ( .D(n5), .CLK(clk_i), 
        .RESET_B(rstn_i), .Q(rst_delay_reg_0) );
  sky130_fd_sc_hd__dfrtp_1 \rst_delay_reg_reg[1]  ( .D(rst_delay_reg_0), .CLK(
        clk_i), .RESET_B(rstn_i), .Q(rstn_o) );
  sky130_fd_sc_hd__dfrtp_1 \rst_delay_reg_reg[2]  ( .D(rstn_o), .CLK(clk_i), 
        .RESET_B(rstn_i), .Q(rst_delay_reg[2]) );
  sky130_fd_sc_hd__dfrtp_1 \rst_delay_reg_reg[3]  ( .D(rst_delay_reg[2]), 
        .CLK(clk_i), .RESET_B(rstn_i), .Q(rst_delay_reg[3]) );
  sky130_fd_sc_hd__conb_1 U3 ( .HI(n5) );
endmodule
