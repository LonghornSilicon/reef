// B10: system / fences. Checklist 3.5, 3.12.
// <base>_u8 / _u16: cycles/unit = (u16 - u8) / 8.
// Note from source (Decode.scala:163,396,723): plain `fence` is NOT in
// isFency(), so it neither waits for the LSU nor ends the group; it only
// dispatches from lane 0. fence.i IS fency: waits for LSU idle, lane 0 alone,
// ends the group, then the LSU flush redirects fetch (SCore.scala:108-111).
#define SETUP "li a5, 7\n"
#define TP(X, name, U) \
  X(name##_u8, SETUP, ST_R8(U)) X(name##_u16, SETUP, ST_R8(U) ST_R8(U))

#define CASES(X)                                                       \
  X(empty, SETUP, "")                                                  \
  TP(X, fence, "fence\n")                                              \
  TP(X, fence_a3, "fence\naddi t0, zero, 1\naddi t1, zero, 1\naddi t2, zero, 1\n") \
  TP(X, lw_fence, "lw t0, 0(s0)\nfence\n")                             \
  TP(X, lw_alone, "lw t0, 0(s0)\n")                                    \
  TP(X, fencei, "fence.i\n")                                           \
  TP(X, fencei_a3, "fence.i\naddi t0, zero, 1\naddi t1, zero, 1\naddi t2, zero, 1\n") \
  TP(X, lw_fencei, "lw t0, 0(s0)\nfence.i\n")                          \
  TP(X, sw_fencei, "sw a5, 0(s0)\nfence.i\n")

#include "st_common.h"
