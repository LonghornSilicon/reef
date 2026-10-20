// B7: ROB full. Checklist 3.10, 5.1.
// Single-shot regions: one long div (dividend 2^30: write-back ~D+34), then K
// independent addi (rd rotates over 12 registers). With an 8-entry ROB, the
// div plus 7 addi fit; addi #8 cannot dispatch until the div retires.
// Report T(K) - T(0): flat for K <= 7, then grows.
#define ST_EXTRA_CLOBBERS , "s2", "s3"
#define SETUP "li a2, 0x40000000\nli a3, 3\n"
#define SETUP_SHORT "li a2, 1\nli a3, 3\n"
#define A(r) "addi " #r ", zero, 1\n"
#define K1 A(t1)
#define K2 K1 A(t2)
#define K4 K2 A(t3) A(t4)
#define K5 K4 A(t5)
#define K6 K5 A(t6)
#define K7 K6 A(a4)
#define K8 K7 A(a5)
#define K9 K8 A(a6)
#define K10 K9 A(a7)
#define K12 K10 A(s1) A(s2)
#define K16 K12 A(t1) A(t2) A(t3) A(t4)
#define K24 K16 A(t5) A(t6) A(a4) A(a5) A(a6) A(a7) A(s1) A(s2)
#define K40 K24 A(t1) A(t2) A(t3) A(t4) A(t5) A(t6) A(a4) A(a5) \
            A(a6) A(a7) A(s1) A(s2) A(t1) A(t2) A(t3) A(t4)
#define DIVL "div t0, a2, a3\n"

#define CASES(X)                                                     \
  X(empty, SETUP, "")                                                \
  X(k0, SETUP, DIVL) X(k2, SETUP, DIVL K2) X(k4, SETUP, DIVL K4)              \
  X(k5, SETUP, DIVL K5) X(k6, SETUP, DIVL K6) X(k7, SETUP, DIVL K7)           \
  X(k8, SETUP, DIVL K8) X(k9, SETUP, DIVL K9) X(k10, SETUP, DIVL K10)         \
  X(k12, SETUP, DIVL K12) X(k16, SETUP, DIVL K16) X(k24, SETUP, DIVL K24)     \
  X(k40, SETUP, DIVL K40)                                               \
  X(short_k0, SETUP_SHORT, DIVL) X(short_k40, SETUP_SHORT, DIVL K40)       \
  X(nodiv_k40, SETUP, K40)

#include "st_common.h"
