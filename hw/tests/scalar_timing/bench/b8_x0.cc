// B8: x0 / nop. Checklist 3.17.
// (1) Throughput, 4-instruction units, <base>_u8 / _u16.
// (2) Prefilled buffer: a div (dividend 256, write-back ~D+12) feeds a
//     dependent addi, so the 8-entry instruction buffer fills behind it with
//     the 7-instruction PATTERN. When the div writes back, the buffered
//     instructions dispatch at the dispatch rate, not the fetch rate.
//     Report T(pf_X) - T(pf_base): the extra dispatch cycles of the pattern.
#define SETUP "li s1, 3\nli a2, 256\nli a3, 3\n"
#define TP(X, name, U) \
  X(name##_u8, SETUP, ST_R8(U)) X(name##_u16, SETUP, ST_R8(U) ST_R8(U))
#define LI(r) "addi " #r ", zero, 1\n"
#define AS(r) "addi " #r ", s1, 1\n"
#define NOP "nop\n"

#define PF(p) "div t4, a2, a3\naddi t5, t4, 0\n" p
#define P_BASE LI(t0) LI(t1) LI(t2) LI(t3) LI(a4) LI(a5) LI(a6)
#define P_NOP7 NOP NOP NOP NOP NOP NOP NOP
#define P_NOP_LI6 NOP LI(t0) LI(t1) LI(t2) LI(t3) LI(a4) LI(a5)
#define P_NOP_AS6 NOP AS(t0) AS(t1) AS(t2) AS(t3) AS(a4) AS(a5)
#define P_LI_NOP_LI5 LI(t0) NOP LI(t1) LI(t2) LI(t3) LI(a4) LI(a5)
#define P_X0W "addi zero, s1, 1\n" AS(t0) AS(t1) AS(t2) AS(t3) AS(a4) AS(a5)

#define CASES(X)                                                    \
  X(empty, SETUP, "")                                               \
  TP(X, nop4, NOP NOP NOP NOP)                                      \
  TP(X, addi4, LI(t0) LI(t1) LI(t2) LI(t3))                         \
  TP(X, nop_li3, NOP LI(t0) LI(t1) LI(t2))                          \
  TP(X, nop_as3, NOP AS(t0) AS(t1) AS(t2))                          \
  TP(X, x0w4, "addi zero, s1, 1\naddi zero, s1, 1\naddi zero, s1, 1\naddi zero, s1, 1\n") \
  X(pf_base, SETUP, PF(P_BASE))                                     \
  X(pf_nop7, SETUP, PF(P_NOP7))                                     \
  X(pf_nop_li6, SETUP, PF(P_NOP_LI6))                               \
  X(pf_nop_as6, SETUP, PF(P_NOP_AS6))                               \
  X(pf_li_nop_li5, SETUP, PF(P_LI_NOP_LI5))                         \
  X(pf_x0w_as6, SETUP, PF(P_X0W))

#include "st_common.h"
