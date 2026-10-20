// B4 (risky part, own ELF so a trap cannot take the rest of B4 down):
// misaligned and line-crossing loads/stores. Checklist Load row
// "Line-crossing / misaligned". If these trap, the test reports it and the
// row stays SRC only.
#define SETUP "li a5, 7\n"
#define TP(X, name, U) \
  X(name##_u8, SETUP, ST_R8(U)) X(name##_u16, SETUP, ST_R8(U) ST_R8(U))
#define LD4(off) "lw t0, " #off "(s0)\nlw t1, " #off "(s0)\nlw t2, " #off "(s0)\nlw t3, " #off "(s0)\n"
#define CASES(X)                                  \
  X(empty, SETUP, "")                             \
  TP(X, lw_al, LD4(0))                            \
  TP(X, lw_mis2, LD4(2))                          \
  TP(X, lw_cross14, LD4(14))                      \
  TP(X, sw_cross14, "sw a5, 14(s0)\nsw a5, 46(s0)\n")
#include "st_common.h"
