// Design Compiler stand-in for the vendored edff.sv (M3). Synthesis tool workaround only.
//
// DC T-2022.03 hits an internal assertion (VER-37) on `q <= INIT` when T is a packed struct that
// contains a packed union and an enum (rvv_backend_div_unit's DIV_RES_t), and then leaves edff
// unresolved. This version stores the value as a bit vector of $bits(T). Same ports, parameters,
// reset value and enable behavior; used only by scripts/synth_dc.sh.
module edff (q, e, d, clk, rst_n
);
  parameter type T = logic;
  parameter T INIT  = T'('0);

  output T        q;

  input  logic    e;
  input  T        d;
  input  logic    clk;
  input  logic    rst_n;

  localparam int W = $bits(T);
  localparam logic [W-1:0] INIT_BITS = INIT;
  logic [W-1:0] q_r;

  always @(posedge clk or negedge rst_n)
    if (!rst_n)   q_r <= INIT_BITS;
    else if (e)   q_r <= d;

  assign q = q_r;

endmodule
