// B6: CSR. Checklist 3.6, 3.11, 3.18, CSR row.
// <base>_u8 / _u16 (cycles/unit = (u16 - u8) / 8), div cases _u4 / _u8.
#define SETUP "li a2, 0x40000000\nli a3, 3\nli a4, 0x3f800000\nfmv.w.x fa0, a4\nfmv.w.x fa1, a4\n"
#define TP(X, name, U) \
  X(name##_u8, SETUP, ST_R8(U)) X(name##_u16, SETUP, ST_R8(U) ST_R8(U))
#define TP4(X, name, U) X(name##_u4, SETUP, ST_R4(U)) X(name##_u8, SETUP, ST_R8(U))

#define CASES(X)                                                          \
  X(empty, SETUP, "")                                                     \
  TP(X, csrr_mscratch, "csrr t0, mscratch\n")                             \
  TP(X, csrw_mscratch, "csrw mscratch, a3\n")                             \
  TP(X, csrr_mcycle, "csrr t0, mcycle\n")                                 \
  TP(X, csrr_a3, "csrr t0, mscratch\naddi t1, zero, 1\naddi t2, zero, 1\naddi t3, zero, 1\n") \
  TP(X, csr_use, "csrr t0, mscratch\naddi t1, t0, 0\n")                   \
  TP(X, a3_csrr, "addi t1, zero, 1\naddi t2, zero, 1\naddi t3, zero, 1\ncsrr t0, mscratch\n") \
  TP4(X, div_alone, "div t0, a2, a3\n")                                   \
  TP4(X, div_csrr, "div t0, a2, a3\ncsrr t1, mscratch\n")                 \
  TP4(X, fadd_csrr, "fadd.s ft0, fa0, fa1\ncsrr t1, mscratch\n")          \
  TP4(X, fadd_alone, "fadd.s ft0, fa0, fa1\n")

#include "st_common.h"
