# Independent stream: 8 x vfredusum.vs per iteration, 250 iterations.
# Measures the vector FP reduction throughput.
.include "common.inc"
  START
  vsetivli zero, 4, e32, m1, ta, ma
  li       t0, 0x3f800000   # 1.0f
  vmv.v.x  v1, t0
  vmv.v.x  v2, t0
  vmv.v.i  v3, 0            # 0.0f
  li       t6, 250
  ROI_BEGIN
  .balign 16            # loop at slot 0 of a 16-byte fetch block
1:
  vfredusum.vs v8, v2, v1
  vfredusum.vs v9, v2, v1
  vfredusum.vs v10, v2, v1
  vfredusum.vs v11, v2, v1
  vfredusum.vs v12, v2, v1
  vfredusum.vs v13, v2, v1
  vfredusum.vs v14, v2, v1
  vfredusum.vs v15, v2, v1
  addi     t6, t6, -1
  bnez     t6, 1b
  ROI_END
  END
