// B9: ALU class sweep. Checklist ALU row, 0.2.
// One ELF per group (-DGROUP_BASE / -DGROUP_ZBB) so an unimplemented Zbb op
// cannot take the RV32I results down with it.
// Dependent chain per op: <op>_u8 / _u24, cycles/inst = (u24 - u8) / 16.
// lui/auipc cannot form a RAW chain; they write the same rd (WAW chain).
#define SETUP "li a2, 5\nli a3, 1\n"
#define TP(X, name, U) \
  X(name##_u8, SETUP, ST_R8(U)) X(name##_u24, SETUP, ST_R8(U) ST_R8(U) ST_R8(U))
#define R(op) #op " a2, a2, a3\n"
#define I(op, imm) #op " a2, a2, " #imm "\n"
#define U1(op) #op " a2, a2\n"

#if defined(GROUP_BASE)
#define CASES(X)                                               \
  X(empty, SETUP, "")                                          \
  TP(X, add, R(add)) TP(X, sub, R(sub)) TP(X, sll, R(sll))     \
  TP(X, srl, R(srl)) TP(X, sra, R(sra)) TP(X, slt, R(slt))     \
  TP(X, sltu, R(sltu)) TP(X, xor_r, R(xor)) TP(X, or_r, R(or))     \
  TP(X, and_r, R(and)) TP(X, addi, I(addi, 1))                   \
  TP(X, slti, I(slti, 1)) TP(X, sltiu, I(sltiu, 1))            \
  TP(X, xori, I(xori, 1)) TP(X, ori, I(ori, 1))                \
  TP(X, andi, I(andi, 1)) TP(X, slli, I(slli, 1))              \
  TP(X, srli, I(srli, 1)) TP(X, srai, I(srai, 1))              \
  TP(X, lui, "lui a2, 1\n") TP(X, auipc, "auipc a2, 0\n")
#elif defined(GROUP_ZBB)
#define CASES(X)                                               \
  X(empty, SETUP, "")                                          \
  TP(X, andn, R(andn)) TP(X, orn, R(orn)) TP(X, xnor, R(xnor)) \
  TP(X, max, R(max)) TP(X, maxu, R(maxu)) TP(X, min, R(min))   \
  TP(X, minu, R(minu)) TP(X, rol, R(rol)) TP(X, ror, R(ror))   \
  TP(X, rori, I(rori, 1)) TP(X, clz, U1(clz))                  \
  TP(X, ctz, U1(ctz)) TP(X, cpop, U1(cpop))                    \
  TP(X, sext_b, U1(sext.b)) TP(X, sext_h, U1(sext.h))          \
  TP(X, zext_h, U1(zext.h)) TP(X, orc_b, U1(orc.b))            \
  TP(X, rev8, U1(rev8))
#else
#error "define GROUP_BASE or GROUP_ZBB"
#endif

#include "st_common.h"
