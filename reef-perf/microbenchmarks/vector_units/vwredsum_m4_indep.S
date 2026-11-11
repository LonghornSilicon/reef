# Independent stream: 6 x vwredsum.vs per iteration, 250 iterations.
# Measures the vector widening reduction throughput, 4 uops per instruction.
.include "common.inc"
  START
  li       t0, 32
  vsetvli  zero, t0, e16, m4, ta, ma
  vmv.v.i  v0, 1
  vmv.v.i  v4, 1
  li       t6, 250
  ROI_BEGIN
  .balign 16            # loop at slot 0 of a 16-byte fetch block
1:
  vwredsum.vs v8, v4, v0
  vwredsum.vs v12, v4, v0
  vwredsum.vs v16, v4, v0
  vwredsum.vs v20, v4, v0
  vwredsum.vs v24, v4, v0
  vwredsum.vs v28, v4, v0
  addi     t6, t6, -1
  bnez     t6, 1b
  ROI_END
  END
