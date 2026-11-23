# Independent stream: 8 x vslidedown.vi per iteration, 250 iterations.
# Measures the vector permute throughput.
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
  vslidedown.vi v8, v2, 1
  vslidedown.vi v9, v2, 1
  vslidedown.vi v10, v2, 1
  vslidedown.vi v11, v2, 1
  vslidedown.vi v12, v2, 1
  vslidedown.vi v13, v2, 1
  vslidedown.vi v14, v2, 1
  vslidedown.vi v15, v2, 1
  addi     t6, t6, -1
  bnez     t6, 1b
  ROI_END
  END
