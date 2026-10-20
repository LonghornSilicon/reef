// TRACE: single-instruction trace for deliverable/timing.json (run with waves).
// Each class sits alone between the two csrr fences:
//     trp_<c>:  addi t5, zero, 1      independent, before
//     tr_<c>:   <instruction under test>
//     tra_<c>:  addi t6, zero, 1      independent, after
//     trc_<c>:  <dependent consumer, or the branch target>   (if any)
// The global labels give the PCs the VCD scanner looks for. Int results go to
// t0, float results to ft0. Int consumer: addi t4, t0, 0. Float consumer:
// fmv.x.w t4, ft0 (an FP op, so it also sees the FPU-busy structural stall).
// LEAD: two lines of independent addi between the start csrr and the traced
// line, so the start fence's ROB-empty wait is absorbed and the traced line
// flows at the normal front-end rate (IBuf -> D in the same cycle).
#define SETUP                                                          \
  "li a2, 5\nli a3, 3\nli a4, 1\nli a5, 0x40000000\n"                  \
  "li a6, 0x3f800000\nfmv.w.x fa0, a6\nli t0, 0x40000000\n"            \
  "fmv.w.x fa1, t0\nfmv.w.x fa2, t0\nfsw fa0, 0(s0)\nsw a2, 4(s0)\n"   \
  "li t3, 2\nvsetivli zero, 4, e32, m1, ta, ma\n"
#define L(p, c) ".global " #p "_" #c "\n" #p "_" #c ":\n"
#define LEAD ST_R2("addi t2, zero, 1\naddi a7, zero, 1\naddi s1, zero, 1\naddi t4, zero, 1\n")
#define PRE(c) LEAD L(trp, c) "addi t5, zero, 1\n"
#define POST(c) L(tra, c) "addi t6, zero, 1\n"
#define T(c, inst, cons) PRE(c) L(tr, c) inst "\n" POST(c) L(trc, c) cons "\n"
#define T0(c, inst) PRE(c) L(tr, c) inst "\n" POST(c)
#define IC "addi t4, t0, 0"
#define FC "fmv.x.w t4, ft0"

#define CASES(X)                                                             \
  X(empty, SETUP, "")                                                        \
  X(alu, SETUP, T(alu, "add t0, a2, a3", IC))                                \
  X(zbb, SETUP, T(zbb, "clz t0, a2", IC))                                    \
  X(lui, SETUP, T(lui, "lui t0, 1", IC))                                     \
  X(auipc, SETUP, T(auipc, "auipc t0, 0", IC))                               \
  X(br_nt_fwd, SETUP, PRE(br_nt_fwd) L(tr, br_nt_fwd) "bne zero, zero, 1f\n" \
    "1:\n" L(trc, br_nt_fwd) "addi t6, zero, 1\n")                           \
  X(br_t_bwd, SETUP, PRE(br_t_bwd) "j 2f\n" L(trc, br_t_bwd)                 \
    "1: addi t3, t3, -1\nj 3f\n2: addi t3, zero, 1\n" L(tr, br_t_bwd)        \
    "bnez t3, 1b\n3:\n" POST(br_t_bwd))                                      \
  X(br_mp_fwd, SETUP, PRE(br_mp_fwd) L(tr, br_mp_fwd) "beq zero, zero, 1f\n" \
    "addi t6, zero, 2\n1:\n" L(trc, br_mp_fwd) "addi t6, zero, 1\n")         \
  X(br_mp_bwd, SETUP, LEAD "1:\n" L(trp, br_mp_bwd) "addi t5, zero, 1\n" L(tr, br_mp_bwd)                 \
    "bne zero, zero, 1b\n" L(trc, br_mp_bwd) "addi t6, zero, 1\n")           \
  X(jal, SETUP, PRE(jal) L(tr, jal) "jal zero, 1f\naddi t6, zero, 2\n1:\n"   \
    L(trc, jal) "addi t6, zero, 1\n")                                        \
  X(jalr, SETUP, LEAD "auipc t1, 0\n" L(trp, jalr) "addi t5, zero, 1\n" L(tr, jalr) "jalr zero, 16(t1)\n" \
    "addi t6, zero, 2\n" L(trc, jalr) "addi t6, zero, 1\n")                  \
  X(jalr_rdy, "lla t1, 2f\n" SETUP, PRE(jalr_rdy) L(tr, jalr_rdy)            \
    "jalr zero, 0(t1)\naddi t6, zero, 2\n2:\n" L(trc, jalr_rdy)              \
    "addi t6, zero, 1\n")                                                    \
  X(csr, SETUP, T(csr, "csrr t0, mscratch", IC))                             \
  X(mul, SETUP, T(mul, "mul t0, a2, a3", IC))                                \
  X(div_small, SETUP, T(div_small, "div t0, a4, a3", IC))                    \
  X(div_large, SETUP, T(div_large, "div t0, a5, a3", IC))                    \
  X(div_by0, SETUP, T(div_by0, "div t0, a5, zero", IC))                      \
  X(lw, SETUP, T(lw, "lw t0, 4(s0)", IC))                                    \
  X(sw, SETUP, T0(sw, "sw a2, 8(s0)"))                                       \
  X(flw, SETUP, T(flw, "flw ft0, 0(s0)", FC))                                \
  X(fsw, SETUP, T0(fsw, "fsw fa0, 12(s0)"))                                  \
  X(fadd, SETUP, T(fadd, "fadd.s ft0, fa0, fa1", FC))                        \
  X(fmul, SETUP, T(fmul, "fmul.s ft0, fa0, fa1", FC))                        \
  X(fmadd, SETUP, T(fmadd, "fmadd.s ft0, fa0, fa1, fa2", FC))                \
  X(fdiv, SETUP, T(fdiv, "fdiv.s ft0, fa0, fa1", FC))                        \
  X(fsqrt, SETUP, T(fsqrt, "fsqrt.s ft0, fa1", FC))                          \
  X(feq, SETUP, T(feq, "feq.s t0, fa0, fa1", IC))                            \
  X(fcvt_w_s, SETUP, T(fcvt_w_s, "fcvt.w.s t0, fa0", IC))                    \
  X(fmv_x_w, SETUP, T(fmv_x_w, "fmv.x.w t0, fa0", IC))                       \
  X(fmv_w_x, SETUP, T(fmv_w_x, "fmv.w.x ft0, a2", FC))                       \
  X(fence, SETUP, T0(fence, "fence"))                                        \
  X(fence_i, SETUP, T0(fence_i, "fence.i"))                                  \
  X(rvv, SETUP, T0(rvv, "vadd.vv v1, v2, v3"))

#include "st_common.h"
