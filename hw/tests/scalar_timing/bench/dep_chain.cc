// Checklist 3.14 (dependent addi chain) + 3.3 (registered scoreboard for store
// data / load base). One source, three ELFs: -DCONSUMER_ALU / _SW / _LW.
//
// A "unit" is
//     P                         addi a2, a2, 0      producer; also loop-carried
//                                                   chain (each P reads the last P)
//     F x d                     addi fN, zero, 1    d independent fillers
//     C                         the consumer under test, reads a2:
//        ALU: addi L, a2, 1     (reads rs1 via the same-cycle `comb` scoreboard)
//        SW : sw   a2, 0(s0)    (store DATA = rs2, registered `regd` scoreboard)
//        LW : lw   L, 0(a2)     (load BASE  = rs1, registered `regd` scoreboard)
// d = number of instructions between P and C (0..3).
//
// Each (d, mode) runs U=8 and U=16 units; cycles/unit = (c16 - c8) / 8, which
// cancels the fixed csrr/pipeline overhead. mode DEP uses a2; mode IND swaps C
// for an instruction of the same type that does NOT read a2 (control).
//
// Leaf destinations rotate over 8 registers (loads stay in flight for several
// cycles, a repeated rd would WAW-stall); fillers alternate between two
// register triples so two units sharing a dispatch group never share an rd.
//
// Regions are line-aligned the same way as fetch_ipc.cc: first csrr in slot 3
// of a 16-byte line, body starts at slot 0.
#include <cstdint>

#define REPS 3
#define NREG 19
// results[rep * NREG + idx]:
//   0            empty body
//   1..16        1 + (d*2 + mode)*2 + size   (mode: 0=DEP 1=IND, size: 0=U8 1=U16)
//   17, 18       chain of 16 / 64 `addi a2,a2,1`
uint32_t results[REPS * NREG];

#define F(r) "addi " #r ", zero, 1\n"
#define P "addi a2, a2, 0\n"

#if defined(CONSUMER_ALU)
#define DEP(L) "addi " #L ", a2, 1\n"
#define IND(L) "addi " #L ", zero, 1\n"
#elif defined(CONSUMER_SW)
#define DEP(L) "sw a2, 0(s0)\n"
#define IND(L) "sw s1, 0(s0)\n"
#elif defined(CONSUMER_LW)
#define DEP(L) "lw " #L ", 0(a2)\n"
#define IND(L) "lw " #L ", 0(s0)\n"
#else
#error "define CONSUMER_ALU, CONSUMER_SW or CONSUMER_LW"
#endif

#define UNIT_D0(K, L, A0, A1, A2) P K(L)
#define UNIT_D1(K, L, A0, A1, A2) P F(A0) K(L)
#define UNIT_D2(K, L, A0, A1, A2) P F(A0) F(A1) K(L)
#define UNIT_D3(K, L, A0, A1, A2) P F(A0) F(A1) F(A2) K(L)

#define EIGHT(U, K)                                                           \
  U(K, t0, s2, s3, s4) U(K, t1, s5, s6, s7) U(K, t2, s2, s3, s4)             \
  U(K, t3, s5, s6, s7) U(K, t4, s2, s3, s4) U(K, t5, s5, s6, s7)             \
  U(K, t6, s2, s3, s4) U(K, a3, s5, s6, s7)
#define U8(D, K) EIGHT(UNIT_D##D, K)
#define U16(D, K) U8(D, K) U8(D, K)

#define CH4 "addi a2, a2, 1\naddi a2, a2, 1\naddi a2, a2, 1\naddi a2, a2, 1\n"
#define CH16 CH4 CH4 CH4 CH4
#define CH64 CH16 CH16 CH16 CH16

#define CLOBBERS                                                              \
  "a0", "a1", "a2", "a3", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s0",     \
      "s1", "s2", "s3", "s4", "s5", "s6", "s7", "memory"

// s0 = scratch pointer (valid DTCM address), a2 = same, s1 = store data for IND.
#define MEASURE(NAME, BODY)                                                   \
  static inline __attribute__((always_inline)) uint32_t NAME(uint32_t* p) {   \
    uint32_t d;                                                               \
    asm volatile("mv s0, %1\n"                                                \
                 "mv a2, %1\n"                                                \
                 "li s1, 5\n"                                                 \
                 ".balign 16\n"                                               \
                 "nop\nnop\nnop\n"                                            \
                 "csrr a0, mcycle\n" BODY "csrr a1, mcycle\n"                 \
                 "sub %0, a1, a0\n"                                           \
                 : "=&r"(d)                                                   \
                 : "r"(p)                                                     \
                 : CLOBBERS);                                                 \
    return d;                                                                 \
  }

#define DEFINE_D(D)                                                           \
  MEASURE(r_d##D##_dep_8, U8(D, DEP))                                         \
  MEASURE(r_d##D##_dep_16, U16(D, DEP))                                       \
  MEASURE(r_d##D##_ind_8, U8(D, IND))                                         \
  MEASURE(r_d##D##_ind_16, U16(D, IND))
DEFINE_D(0)
DEFINE_D(1)
DEFINE_D(2)
DEFINE_D(3)
MEASURE(r_empty, "")
MEASURE(r_chain16, CH16)
MEASURE(r_chain64, CH64)

static uint32_t scratch[8];

#define STORE_D(D)                                                            \
  r[1 + (D * 2 + 0) * 2 + 0] = r_d##D##_dep_8(scratch);                       \
  r[1 + (D * 2 + 0) * 2 + 1] = r_d##D##_dep_16(scratch);                      \
  r[1 + (D * 2 + 1) * 2 + 0] = r_d##D##_ind_8(scratch);                       \
  r[1 + (D * 2 + 1) * 2 + 1] = r_d##D##_ind_16(scratch);

int main(void) {
#pragma GCC unroll 1
  for (int rep = 0; rep < REPS; rep++) {
    uint32_t* r = &results[rep * NREG];
    r[0] = r_empty(scratch);
    STORE_D(0)
    STORE_D(1)
    STORE_D(2)
    STORE_D(3)
    r[17] = r_chain16(scratch);
    r[18] = r_chain64(scratch);
  }
  return 0;
}
