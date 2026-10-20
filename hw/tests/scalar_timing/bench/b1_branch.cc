// B1: branches. Checklist 1.7, 1.8, 1.10, 3.4, 3.5, BRU rows.
// Throughput cases are <base>_u8 / <base>_u16; cycles/unit = (u16 - u8) / 8
// (joff*: two-line units, <base>_u4 / _u8, cycles/unit = (u8 - u4) / 4).
// Loops are <base>_n8 / <base>_n16 iterations; cycles/iter = (n16 - n8) / 8.
// Every unit is a whole number of 16-byte lines, so alignment is preserved.
#define A(r) "addi " #r ", zero, 1\n"

// 4 not-taken forward branches per line (target = next instruction).
#define U_BNT "bne zero, zero, 1f\n1:\n"
// Not-taken branch in lane 0 followed by 3 independent addi.
#define U_BNT_A3 "bne zero, zero, 1f\n1:\n" A(t0) A(t1) A(t2)
// Same, branch in lane 3.
#define U_A3_BNT A(t0) A(t1) A(t2) "bne zero, zero, 1f\n1:\n"
// Forward taken (BTFN predicts not-taken -> mispredict), target = next line.
#define U_FT "beq zero, zero, 1f\n" A(t0) A(t1) A(t2) "1:\n"
// Backward not-taken (BTFN predicts taken -> mispredict); branch in slot 3, so
// the fall-through is the next line.
#define U_BNT_BWD A(t0) A(t1) "1:\n" A(t2) "bne zero, zero, 1b\n"
// jal (always predicted), target = next line.
#define U_JAL "jal zero, 1f\n" A(t0) A(t1) A(t2) "1:\n"
// jalr is never predicted. auipc -> jalr (rs1 via registered scoreboard).
#define U_JALR "auipc t3, 0\njalr zero, 16(t3)\n" A(t0) A(t1) "1:\n"
// Predicted jal into the middle of the next line (1.10): k skipped slots.
#define U_JOFF0 "jal zero, 1f\n" A(t0) A(t1) A(t2) "1:\n" A(t4) A(t5) A(t6) A(a2)
#define U_JOFF1 "jal zero, 1f\n" A(t0) A(t1) A(t2) A(t3) "1:\n" A(t4) A(t5) A(t6)
#define U_JOFF2 "jal zero, 1f\n" A(t0) A(t1) A(t2) A(t3) A(t3) "1:\n" A(t4) A(t5)
#define U_JOFF3 "jal zero, 1f\n" A(t0) A(t1) A(t2) A(t3) A(t3) A(t3) "1:\n" A(t4)
// Plain addi line, the fetch-bound reference.
#define U_A4 A(t0) A(t1) A(t2) A(t3)

#define TP(X, name, U) \
  X(name##_u8, "", ST_R8(U)) X(name##_u16, "", ST_R8(U) ST_R8(U))
// Two-line units: 4 and 8 units, cycles/unit = (u8 - u4) / 4.
#define TP4(X, name, U) X(name##_u4, "", ST_R4(U)) X(name##_u8, "", ST_R8(U))

// Backward-taken loops (predicted at predecode). Body starts line-aligned.
#define LOOP2(n) "li t3, " #n "\n", "1: addi t3, t3, -1\nbnez t3, 1b\n"
#define LOOP4(n) "li t3, " #n "\n", \
  "1: addi t0, zero, 1\naddi t1, zero, 1\naddi t3, t3, -1\nbnez t3, 1b\n"
#define LOOP8(n) "li t3, " #n "\n", \
  "1:\n" A(t0) A(t1) A(t2) A(t4) A(t5) A(t6) "addi t3, t3, -1\nbnez t3, 1b\n"

#define CASES(X)                                       \
  X(empty, "", "")                                     \
  TP(X, a4, U_A4)                                         \
  TP(X, bnt, U_BNT)                                       \
  TP(X, bnt_a3, U_BNT_A3)                                 \
  TP(X, a3_bnt, U_A3_BNT)                                 \
  TP(X, ft, U_FT)                                         \
  TP(X, bnt_bwd, U_BNT_BWD)                               \
  TP(X, jal, U_JAL)                                       \
  TP(X, jalr, U_JALR)                                     \
  TP4(X, joff0, U_JOFF0)                                   \
  TP4(X, joff1, U_JOFF1)                                   \
  TP4(X, joff2, U_JOFF2)                                   \
  TP4(X, joff3, U_JOFF3)                                   \
  ST_CALL(X, loop2_n8, LOOP2(8)) ST_CALL(X, loop2_n16, LOOP2(16))        \
  ST_CALL(X, loop4_n8, LOOP4(8)) ST_CALL(X, loop4_n16, LOOP4(16))        \
  ST_CALL(X, loop8_n8, LOOP8(8)) ST_CALL(X, loop8_n16, LOOP8(16))

#include "st_common.h"
