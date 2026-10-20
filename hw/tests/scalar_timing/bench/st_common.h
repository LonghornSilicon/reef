// Shared harness for the sprint benches (B1-B10, TRACE).
//
// Each bench .cc defines an X-macro list
//     #define CASES(X) X(name, SETUP, BODY) X(...) ...
// and then includes this header. Every case becomes one noinline function:
//
//     la s0, st_scratch ; SETUP           <- not timed
//     .balign 16 ; nop nop nop             <- first csrr = slot 3 of a line
//     csrr a0, mcycle                      <- start (waits for ROB empty)
//     BODY                                 <- starts at slot 0 of a fresh line
//     csrr a1, mcycle                      <- end
//     sub  <out>, a1, a0
//
// Register rules for SETUP/BODY: a0/a1 hold the timestamps, s0 = &st_scratch,
// the result goes to a compiler-chosen register after the second csrr.
// Free for bodies: t0-t6, a2-a7, s1, ft0-ft11, fa0-fa7 (plus whatever
// ST_EXTRA_CLOBBERS adds). Never write ra/sp/gp/tp, a0, a1, s0.
//
// The ELF exports
//     st_results[REPS * ST_NCASES]   raw (a1 - a0) per rep and case
//     st_case_names                  "name0,name1,..." (same order)
//     st_misa                        misa CSR read at the start of main
// and the cocotb harness maps results to names, so the Python side never
// depends on case order.
#pragma once
#include <cstdint>

#ifndef REPS
#define REPS 2
#endif

// Repetition helpers.
#define ST_R2(x) x x
#define ST_R4(x) ST_R2(x) ST_R2(x)
#define ST_R8(x) ST_R4(x) ST_R4(x)
// ST_CALL(X, name, M(...)) lets M expand to "SETUP", "BODY" before X sees it.
#define ST_CALL(X, ...) X(__VA_ARGS__)

// Callee-saved registers are NOT clobbered by default (each one costs a
// save/restore in every region's prologue, and ITCM is only 8 KB). A bench
// that needs more registers defines ST_EXTRA_CLOBBERS as  , "s2", "s3", ...
#ifndef ST_EXTRA_CLOBBERS
#define ST_EXTRA_CLOBBERS
#endif
#define ST_CLOBBERS                                                           \
  "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a0", "a1", "a2", "a3", "a4",     \
      "a5", "a6", "a7", "s0", "s1", "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", \
      "ft6", "ft7", "ft8", "ft9", "ft10", "ft11", "fa0", "fa1", "fa2", "fa3", \
      "fa4", "fa5", "fa6", "fa7", "memory" ST_EXTRA_CLOBBERS

__attribute__((used, aligned(64))) uint32_t st_scratch[64];
__attribute__((used)) uint32_t st_misa;

#define ST_REGION(NAME, SETUP, BODY)                                          \
  static __attribute__((noinline)) uint32_t r_##NAME() {                      \
    uint32_t d;                                                               \
    asm volatile("la s0, st_scratch\n" SETUP                                  \
                 "\n.balign 16\nnop\nnop\nnop\n"                              \
                 "csrr a0, mcycle\n" BODY "\ncsrr a1, mcycle\n"               \
                 "sub %0, a1, a0\n"                                           \
                 : "=r"(d)                                                    \
                 :                                                            \
                 : ST_CLOBBERS);                                              \
    return d;                                                                 \
  }

#define ST_X_FN(NAME, SETUP, BODY) ST_REGION(NAME, SETUP, BODY)
#define ST_X_COUNT(NAME, SETUP, BODY) +1
#define ST_X_NAME(NAME, SETUP, BODY) #NAME ","
#define ST_X_RUN(NAME, SETUP, BODY) r[i++] = r_##NAME();

CASES(ST_X_FN)
constexpr int ST_NCASES = 0 CASES(ST_X_COUNT);
__attribute__((used)) uint32_t st_results[REPS * ST_NCASES];
__attribute__((used)) char st_case_names[] = CASES(ST_X_NAME);

int main(void) {
  uint32_t misa;
  asm volatile("csrr %0, misa" : "=r"(misa));
  st_misa = misa;
  // Keep the name table alive through --gc-sections (only cocotb reads it).
  asm volatile("" ::"r"(st_case_names) : "memory");
#pragma GCC unroll 1
  for (int rep = 0; rep < REPS; rep++) {
    uint32_t* r = &st_results[rep * ST_NCASES];
    int i = 0;
    CASES(ST_X_RUN)
  }
  return 0;
}

