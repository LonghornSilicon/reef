// Inner-loop microbenchmark for conv2d.cc: cycles per k-step of variants.
//   -DM=1 loads only (8 x vle8), -DM=2 loads+unpack, -DM=3 loads+vtmm,
//   -DM=4 loads+unpack+vtmm.  -DMIS=1 misaligned addresses.
#include "ops.hpp"

using namespace ops;
#ifndef M
#define M 4
#endif
#ifndef MIS
#define MIS 0
#endif
static uint8_t buf[8192] __attribute__((aligned(16)));

int main() {
    matrix_on();
    vtzero(true);
    config_int8(16);
    const uint8_t* a = buf + MIS;
    const uint8_t* b = buf + 4096;
    uint32_t steps = 1000;
    const uint32_t stride = 0; // re-read the same lines
    roi_begin();
#if M == 5 || M == 6
    // loads into odd temps + 16 independent ALU ops (+ 4 vtmm for M=6)
    asm volatile(
        "1:\n\t"
        "vle8.v v1, (%0)\n\t" "vsll.vi v8, v9, 4\n\t" "vand.vi v10, v9, -16\n\t"
        "vle8.v v3, (%1)\n\t" "vsll.vi v16, v11, 4\n\t" "vand.vi v18, v11, -16\n\t"
        "vle8.v v5, (%0)\n\t" "vsll.vi v12, v13, 4\n\t" "vand.vi v14, v13, -16\n\t"
        "vle8.v v7, (%1)\n\t" "vsll.vi v20, v15, 4\n\t" "vand.vi v22, v15, -16\n\t"
#if M == 6
        ".insn r 0x77, 0, 0x79, x1, x16, x8\n\t"
#endif
        "vle8.v v9, (%1)\n\t" "vsll.vi v0, v17, 4\n\t" "vand.vi v2, v17, -16\n\t"
        "vle8.v v11, (%1)\n\t" "vsll.vi v4, v19, 4\n\t" "vand.vi v6, v19, -16\n\t"
#if M == 6
        ".insn r 0x77, 0, 0x79, x9, x0, x8\n\t"
#endif
        "vle8.v v13, (%0)\n\t" "vsll.vi v24, v21, 4\n\t" "vand.vi v26, v21, -16\n\t"
        "vle8.v v15, (%0)\n\t" "vsll.vi v28, v23, 4\n\t" "vand.vi v30, v23, -16\n\t"
#if M == 6
        ".insn r 0x77, 0, 0x79, x17, x16, x24\n\t"
        ".insn r 0x77, 0, 0x79, x25, x0, x24\n\t"
#endif
        "add %0, %0, %3\n\taddi %2, %2, -1\n\tbnez %2, 1b"
        : "+r"(a), "+r"(b), "+r"(steps)
        : "r"(stride)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v26", "v28", "v30", "memory");
#else
    asm volatile(
        "1:\n\t"
        "vle8.v v10, (%0)\n\t"
        "vle8.v v18, (%1)\n\t"
        "vle8.v v14, (%0)\n\t"
        "vle8.v v22, (%1)\n\t"
#if M == 2 || M == 4
        "vsll.vi v8, v10, 4\n\tvand.vi v10, v10, -16\n\t"
        "vsll.vi v16, v18, 4\n\tvand.vi v18, v18, -16\n\t"
        "vsll.vi v12, v14, 4\n\tvand.vi v14, v14, -16\n\t"
        "vsll.vi v20, v22, 4\n\tvand.vi v22, v22, -16\n\t"
#endif
        "vle8.v v2, (%1)\n\t"
        "vle8.v v6, (%1)\n\t"
#if M >= 3
        ".insn r 0x77, 0, 0x79, x1, x16, x8\n\t"
#endif
        "vle8.v v26, (%0)\n\t"
        "vle8.v v30, (%0)\n\t"
#if M == 2 || M == 4
        "vsll.vi v0, v2, 4\n\tvand.vi v2, v2, -16\n\t"
        "vsll.vi v4, v6, 4\n\tvand.vi v6, v6, -16\n\t"
#endif
#if M >= 3
        ".insn r 0x77, 0, 0x79, x9, x0, x8\n\t"
#endif
#if M == 2 || M == 4
        "vsll.vi v24, v26, 4\n\tvand.vi v26, v26, -16\n\t"
        "vsll.vi v28, v30, 4\n\tvand.vi v30, v30, -16\n\t"
#endif
#if M >= 3
        ".insn r 0x77, 0, 0x79, x17, x16, x24\n\t"
        ".insn r 0x77, 0, 0x79, x25, x0, x24\n\t"
#endif
        "add %0, %0, %3\n\taddi %2, %2, -1\n\tbnez %2, 1b"
        : "+r"(a), "+r"(b), "+r"(steps)
        : "r"(stride)
        : "v0", "v2", "v4", "v6", "v8", "v10", "v12", "v14", "v16", "v18", "v20", "v22",
          "v24", "v26", "v28", "v30", "memory");
#endif
    roi_end();
    return 0;
}
