// B5: FPU. Checklist 3.6, 3.16, FPU rows.
// Units hold 2 FP ops; <base>_u4 / _u8, cycles/op = (u8 - u4) / 4 / 2.
// fa0 = 1.0, fa1 = 2.0, fa2 = 2.0, fa3 = 0.0, a2 = 0x3f800000.
#define SETUP                                                    \
  "li a2, 0x3f800000\nfmv.w.x fa0, a2\nli t0, 0x40000000\n"      \
  "fmv.w.x fa1, t0\nfmv.w.x fa2, t0\nfmv.w.x fa3, zero\n"        \
  "fsw fa0, 0(s0)\nfsw fa0, 4(s0)\n"
#define TP(X, name, U) X(name##_u4, SETUP, ST_R4(U)) X(name##_u8, SETUP, ST_R8(U))

#define U_FADD "fadd.s ft0, fa0, fa1\nfadd.s ft1, fa0, fa1\n"
#define U_FADD_DEP "fadd.s fa2, fa2, fa3\nfadd.s fa2, fa2, fa3\n"
#define U_FMUL "fmul.s ft0, fa0, fa1\nfmul.s ft1, fa0, fa1\n"
#define U_FMADD "fmadd.s ft0, fa0, fa1, fa2\nfmadd.s ft1, fa0, fa1, fa2\n"
#define U_FDIV "fdiv.s ft0, fa0, fa1\nfdiv.s ft1, fa0, fa1\n"
#define U_FSQRT "fsqrt.s ft0, fa1\nfsqrt.s ft1, fa1\n"
#define U_FEQ "feq.s t0, fa0, fa1\nfeq.s t1, fa0, fa1\n"
#define U_FCVT "fcvt.w.s t0, fa0\nfcvt.w.s t1, fa0\n"
#define U_FMVXW "fmv.x.w t0, fa0\nfmv.x.w t1, fa0\n"
#define U_FMVWX "fmv.w.x ft0, a2\nfmv.w.x ft1, a2\n"
#define U_FLW "flw ft0, 0(s0)\nflw ft1, 4(s0)\n"
#define U_FSW "fsw fa0, 8(s0)\nfsw fa0, 12(s0)\n"
// One FP op per unit (cycles/unit = (u8 - u4) / 4):
#define U_FADD_A3 "fadd.s ft0, fa0, fa1\naddi t0, zero, 1\naddi t1, zero, 1\naddi t2, zero, 1\n"
#define U_FCVT_USE "fcvt.w.s t0, fa0\naddi t1, t0, 0\n"
#define U_FMVWX_FADD "fmv.w.x ft0, a2\nfadd.s ft1, ft0, fa1\n"
#define U_FLW_USE "flw ft0, 0(s0)\nfmv.x.w t0, ft0\n"

#define CASES(X)                         \
  X(empty, SETUP, "")                    \
  TP(X, fadd, U_FADD)                    \
  TP(X, fadd_dep, U_FADD_DEP)            \
  TP(X, fmul, U_FMUL)                    \
  TP(X, fmadd, U_FMADD)                  \
  TP(X, fdiv, U_FDIV)                    \
  TP(X, fsqrt, U_FSQRT)                  \
  TP(X, feq, U_FEQ)                      \
  TP(X, fcvt, U_FCVT)                    \
  TP(X, fmvxw, U_FMVXW)                  \
  TP(X, fmvwx, U_FMVWX)                  \
  TP(X, flw, U_FLW)                      \
  TP(X, fsw, U_FSW)                      \
  TP(X, fadd_a3, U_FADD_A3)              \
  TP(X, fcvt_use, U_FCVT_USE)            \
  TP(X, fmvwx_fadd, U_FMVWX_FADD)        \
  TP(X, flw_use, U_FLW_USE)

#include "st_common.h"
