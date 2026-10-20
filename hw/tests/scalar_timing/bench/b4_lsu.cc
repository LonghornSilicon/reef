// B4: LSU latency and structure. Checklist 3.9, 3.19, Load/Store rows.
// <base>_u8 / _u16: cycles/unit = (u16 - u8) / 8 (byte/half variants: _u4 / _u8, / 4).
// st_scratch[0] = &st_scratch[0] (pointer chase), a2 = &st_scratch, a5 = data.
#define SETUP "sw s0, 0(s0)\nmv a2, s0\nli a5, 7\n"
#define TP(X, name, U) \
  X(name##_u8, SETUP, ST_R8(U)) X(name##_u16, SETUP, ST_R8(U) ST_R8(U))
// Smaller variant (ITCM is 8 KB): cycles/unit = (u8 - u4) / 4.
#define TP4(X, name, U) X(name##_u4, SETUP, ST_R4(U)) X(name##_u8, SETUP, ST_R8(U))

// Load-to-use: pointer chase (load base via registered scoreboard).
#define U_CHASE "lw a2, 0(a2)\n"
// Load -> ALU -> next load base.
#define U_CHASE_ALU "lw a4, 0(a2)\nadd a2, a4, zero\n"
// Independent streams, rd rotates over 4.
#define LD4(op) op " t0, 0(s0)\n" op " t1, 4(s0)\n" op " t2, 8(s0)\n" op " t3, 12(s0)\n"
#define ST4(op) op " a5, 16(s0)\n" op " a5, 20(s0)\n" op " a5, 24(s0)\n" op " a5, 28(s0)\n"
// Store then load, same / different word (single LSU slot).
#define U_SL_SAME "sw a5, 32(s0)\nlw t0, 32(s0)\n"
#define U_SL_DIFF "sw a5, 32(s0)\nlw t0, 36(s0)\n"
#define U_LS "lw t0, 0(s0)\nsw a5, 40(s0)\n"
// Load -> dependent store data (store data via registered scoreboard).
#define U_L_SDATA "lw t0, 0(s0)\nsw t0, 44(s0)\n"

// Single-shot bursts of n independent loads (opQueue = 4 entries).
#define LD1 "lw t0, 0(s0)\n"
#define LD2 LD1 "lw t1, 4(s0)\n"
#define LD3 LD2 "lw t2, 8(s0)\n"
#define LD4B LD3 "lw t3, 12(s0)\n"
#define LD5 LD4B "lw t4, 16(s0)\n"
#define LD6 LD5 "lw t5, 20(s0)\n"
#define LD8 LD6 "lw t6, 24(s0)\nlw a3, 28(s0)\n"

#define CASES(X)                                  \
  X(empty, SETUP, "")                             \
  TP(X, chase, U_CHASE)                           \
  TP(X, chase_alu, U_CHASE_ALU)                   \
  TP(X, lw4, LD4("lw"))                           \
  TP4(X, lh4, LD4("lh"))                           \
  TP4(X, lhu4, LD4("lhu"))                         \
  TP4(X, lb4, LD4("lb"))                           \
  TP4(X, lbu4, LD4("lbu"))                         \
  TP(X, sw4, ST4("sw"))                           \
  TP4(X, sh4, ST4("sh"))                           \
  TP4(X, sb4, ST4("sb"))                           \
  TP(X, sl_same, U_SL_SAME)                       \
  TP(X, sl_diff, U_SL_DIFF)                       \
  TP(X, ls, U_LS)                                 \
  TP(X, l_sdata, U_L_SDATA)                       \
  X(burst1, SETUP, LD1) X(burst2, SETUP, LD2)     \
  X(burst3, SETUP, LD3) X(burst4, SETUP, LD4B)    \
  X(burst5, SETUP, LD5) X(burst6, SETUP, LD6)     \
  X(burst8, SETUP, LD8)

#include "st_common.h"
