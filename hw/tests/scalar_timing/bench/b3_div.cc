// B3: DVU. Checklist 3.7, 3.15, DVU row.
// Independent divides, rd rotates: <base>_u4 / _u8, cycles/div = (u8 - u4) / 4.
// a2 = dividend, a3 = divisor (3, or 0 for divide-by-zero).
#define DSETUP(a, b) "li a2, " #a "\nli a3, " #b "\n"
#define DIV4(op) op " t0, a2, a3\n" op " t1, a2, a3\n" op " t2, a2, a3\n" op " t3, a2, a3\n"
#define TPD(X, name, a, b, op) \
  X(name##_u4, DSETUP(a, b), DIV4(op)) X(name##_u8, DSETUP(a, b), DIV4(op) DIV4(op))

// Dependent: div -> and -> or back into the dividend (value unchanged).
// Period = (div write-back) + 2 ALU hops.
#define U_DDEP "div a4, a2, a3\nand a5, a4, zero\nor a2, a2, a5\n"
#define TPDEP(X, name, a)                                              \
  X(name##_u4, DSETUP(a, 3), ST_R4(U_DDEP))                            \
  X(name##_u8, DSETUP(a, 3), ST_R8(U_DDEP))

// Port-4 collision: div (dividend 1, N=2) followed by 6 independent muls vs
// 6 independent addi (control). MLU has priority on write port 4.
#define U_DMUL "div t0, a2, a3\nmul t1, a3, a3\nmul t2, a3, a3\nmul t3, a3, a3\n" \
               "mul t4, a3, a3\nmul t5, a3, a3\nmul t6, a3, a3\n"
#define U_DADD "div t0, a2, a3\naddi t1, zero, 1\naddi t2, zero, 1\naddi t3, zero, 1\n" \
               "addi t4, zero, 1\naddi t5, zero, 1\naddi t6, zero, 1\n"

#define CASES(X)                                       \
  X(empty, DSETUP(1, 3), "")                           \
  TPD(X, div_a1, 1, 3, "div")                          \
  TPD(X, div_a0, 0, 3, "div")                          \
  TPD(X, div_a256, 256, 3, "div")                      \
  TPD(X, div_a65536, 65536, 3, "div")                  \
  TPD(X, div_a2p30, 0x40000000, 3, "div")              \
  TPD(X, div_a2p31, 0x80000000, 3, "div")              \
  TPD(X, div_am1, -1, 3, "div")                        \
  TPD(X, div_by0, 256, 0, "div")                       \
  TPD(X, divu_a256, 256, 3, "divu")                    \
  TPD(X, rem_a256, 256, 3, "rem")                      \
  TPD(X, remu_a256, 256, 3, "remu")                    \
  TPD(X, divu_a2p31, 0x80000000, 3, "divu")            \
  TPDEP(X, ddep_a256, 256)                             \
  TPDEP(X, ddep_a1, 1)                                 \
  X(dmul_u4, DSETUP(1, 3), ST_R4(U_DMUL))              \
  X(dmul_u8, DSETUP(1, 3), ST_R8(U_DMUL))              \
  X(dadd_u4, DSETUP(1, 3), ST_R4(U_DADD))              \
  X(dadd_u8, DSETUP(1, 3), ST_R8(U_DADD))

#include "st_common.h"
