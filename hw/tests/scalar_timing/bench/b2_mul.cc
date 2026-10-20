// B2: MLU. Checklist 3.8, 3.15, MLU row.
// <base>_u8 / _u16: cycles/unit = (u16 - u8) / 8.
// a2 = chain register, a3 = 1 (so the chain value stays put), a4 = 3.
#define SETUP "li a2, 5\nli a3, 1\nli a4, 3\n"
#define TP(X, name, U) \
  X(name##_u8, SETUP, ST_R8(U)) X(name##_u16, SETUP, ST_R8(U) ST_R8(U))

// Dependent chains (latency to a dependent MLU op).
#define U_MUL "mul a2, a2, a3\n"
#define U_MULH "mulh a2, a2, a3\nori a2, a2, 5\n"
#define U_MULHU "mulhu a2, a2, a3\nori a2, a2, 5\n"
#define U_MULHSU "mulhsu a2, a2, a3\nori a2, a2, 5\n"
// mulh of (5 * 1) is 0, so a dependent `ori` restores the value; the unit is
// mulh -> ori -> next mulh, i.e. one MLU latency + one ALU latency.
#define U_MUL_ORI "mul a2, a2, a3\nori a2, a2, 5\n"
// mul -> dependent addi (ALU consumer).
#define U_MUL_ALU "mul a2, a2, a3\naddi a2, a2, 0\n"
// Independent muls, rd rotates (4 per line).
#define U_MUL_IND "mul t0, a4, a4\nmul t1, a4, a4\nmul t2, a4, a4\nmul t3, a4, a4\n"
// 2 muls + 2 addi per line (structural: one MLU dispatch per cycle).
#define U_MUL2_A2 "mul t0, a4, a4\nmul t1, a4, a4\naddi t2, zero, 1\naddi t3, zero, 1\n"

#define CASES(X)                     \
  X(empty, SETUP, "")                \
  TP(X, mul_chain, U_MUL)            \
  TP(X, mul_ori, U_MUL_ORI)          \
  TP(X, mulh_ori, U_MULH)            \
  TP(X, mulhu_ori, U_MULHU)          \
  TP(X, mulhsu_ori, U_MULHSU)        \
  TP(X, mul_alu, U_MUL_ALU)          \
  TP(X, mul_ind4, U_MUL_IND)         \
  TP(X, mul2_a2, U_MUL2_A2)

#include "st_common.h"
