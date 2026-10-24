# Independent stream: 8 x vmul.vv per iteration, 250 iterations.
# Measures the vector multiplier throughput.
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
  vmul.vv  v8, v1, v2
  vmul.vv  v9, v1, v2
  vmul.vv  v10, v1, v2
  vmul.vv  v11, v1, v2
  vmul.vv  v12, v1, v2
  vmul.vv  v13, v1, v2
  vmul.vv  v14, v1, v2
  vmul.vv  v15, v1, v2
  addi     t6, t6, -1
  bnez     t6, 1b
  ROI_END
  END
