// Microbenchmark for the dwconv design: vector instruction throughput on
// this core (every vector op holds a scalar ROB entry until it retires).
//   -DTEST=1  independent vwmacc.vv e8m1 -> e16m2 (4 accumulators)
//   -DTEST=2  independent vwmacc.vx e8m4 -> e16m8 (2 accumulators)
//   -DTEST=3  vle8 m1 stream          -DTEST=4  vsll.vi m1 stream
//   -DTEST=5  dependent vwmacc chain e8m1 (1 accumulator)
#include "ops.hpp"
using namespace ops;
#ifndef TEST
#define TEST 1
#endif
static uint8_t buf[1024] __attribute__((used));
#define R8(x) x x x x x x x x
int main() {
    uint32_t n = 1000;
    roi_begin();
#if TEST == 2
    asm volatile("vsetivli zero, 16, e8, m4, ta, ma" ::: "vl", "vtype");
#else
    asm volatile("vsetivli zero, 16, e8, m1, ta, ma" ::: "vl", "vtype");
#endif
    for (uint32_t i = 0; i < n; ++i) {
#if TEST == 1
        asm volatile(R8("vwmacc.vv v8, v1, v2\n\tvwmacc.vv v10, v1, v2\n\tvwmacc.vv v12, v1, v2\n\tvwmacc.vv v14, v1, v2\n\t") ::: "v8","v9","v10","v11","v12","v13","v14","v15");
#elif TEST == 2
        asm volatile(R8("vwmacc.vx v8, %0, v4\n\tvwmacc.vx v16, %0, v4\n\t") :: "r"(3) : "v8","v16");
#elif TEST == 3
        asm volatile(R8("vle8.v v1, (%0)\n\tvle8.v v2, (%0)\n\tvle8.v v3, (%0)\n\tvle8.v v4, (%0)\n\t") :: "r"(buf) : "v1","v2","v3","v4");
#elif TEST == 4
        asm volatile(R8("vsll.vi v1, v5, 4\n\tvsll.vi v2, v5, 4\n\tvsll.vi v3, v5, 4\n\tvsll.vi v4, v5, 4\n\t") ::: "v1","v2","v3","v4");
#elif TEST == 5
        asm volatile(R8("vwmacc.vv v8, v1, v2\n\tvwmacc.vv v8, v1, v2\n\tvwmacc.vv v8, v1, v2\n\tvwmacc.vv v8, v1, v2\n\t") ::: "v8","v9");
#endif
    }
    roi_end();
    return 0;
}
