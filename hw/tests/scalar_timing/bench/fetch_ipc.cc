// Checklist 1.5: front-end throughput of independent addi.
//
// Method: every measured region is one big inline-asm block so the compiler
// can't put anything between the two `csrr mcycle` reads.
//
//   .balign 16 ; nop nop nop     <- pad so the first csrr is the LAST slot of a
//                                   16-byte fetch line (the pads run BEFORE the
//                                   first timestamp, so they aren't measured)
//   csrr a0, mcycle              <- start (waits for the ROB to drain, so it
//                                   acts as a fence)
//   <N x addi rd, zero, 1>       <- body starts at slot 0 of a fresh line
//   csrr a1, mcycle              <- end (waits for the body to retire)
//
// All addi read x0, so there are no RAW deps. rd rotates through 16 different
// registers, so there is no WAW within 16 instructions.
//
// body_cycles = (a1 - a0) - (a1 - a0 of the empty body)
#include <cstdint>

#define REPS 3
#define NCASES 4  // empty, 16, 64, 256

// results[rep * NCASES + case]; read back by cocotb via the ELF symbol.
uint32_t results[REPS * NCASES];

#define ADDI(r) "addi " #r ", zero, 1\n"
#define B16                                                                  \
  ADDI(t0) ADDI(t1) ADDI(t2) ADDI(t3) ADDI(t4) ADDI(t5) ADDI(t6) ADDI(a2)    \
  ADDI(a3) ADDI(a4) ADDI(a5) ADDI(a6) ADDI(a7) ADDI(s2) ADDI(s3) ADDI(s4)
#define B64 B16 B16 B16 B16
#define B256 B64 B64 B64 B64

#define CLOBBERS                                                            \
  "a0", "a1", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a2", "a3", "a4",   \
      "a5", "a6", "a7", "s2", "s3", "s4"

#define MEASURE(NAME, BODY)                                                  \
  static inline __attribute__((always_inline)) uint32_t NAME() {             \
    uint32_t d;                                                              \
    asm volatile(".balign 16\n"                                              \
                 "nop\nnop\nnop\n"                                           \
                 "csrr a0, mcycle\n" BODY "csrr a1, mcycle\n"                \
                 "sub %0, a1, a0\n"                                          \
                 : "=r"(d)::CLOBBERS);                                       \
    return d;                                                                \
  }

MEASURE(run_empty, "")
MEASURE(run_16, B16)
MEASURE(run_64, B64)
MEASURE(run_256, B256)

int main(void) {
  for (int rep = 0; rep < REPS; rep++) {
    results[rep * NCASES + 0] = run_empty();
    results[rep * NCASES + 1] = run_16();
    results[rep * NCASES + 2] = run_64();
    results[rep * NCASES + 3] = run_256();
  }
  return 0;
}
