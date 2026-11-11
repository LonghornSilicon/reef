# Independent stream: 8 x vcpop.m per iteration, 250 iterations.
# Measures the vector vcpop.m (permute/reduction unit) to a scalar register.
.include "common.inc"
  START
  vsetivli zero, 4, e32, m1, ta, ma
  vmv.v.i  v1, 1
  vmv.v.i  v2, 1
  vmv.v.i  v3, 0
  li       t6, 250
  ROI_BEGIN
  .balign 16            # loop at slot 0 of a 16-byte fetch block
1:
  vcpop.m  a0, v2
  vcpop.m  a1, v2
  vcpop.m  a2, v2
  vcpop.m  a3, v2
  vcpop.m  a4, v2
  vcpop.m  a5, v2
  vcpop.m  a6, v2
  vcpop.m  a7, v2
  addi     t6, t6, -1
  bnez     t6, 1b
  ROI_END
  END
