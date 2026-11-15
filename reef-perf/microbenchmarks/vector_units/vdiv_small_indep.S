# Independent stream: 8 x vdiv.vv per iteration, 250 iterations.
# Measures the vector divider throughput, 1- and 2-bit dividends.
.include "common.inc"
  START
  vsetivli zero, 4, e32, m1, ta, ma
  vmv.v.i  v1, 1
  vmv.v.i  v4, 2
  vmv.v.i  v2, -1
  li       t6, 250
  ROI_BEGIN
  .balign 16            # loop at slot 0 of a 16-byte fetch block
1:
  vdiv.vv  v8, v1, v2
  vdiv.vv  v9, v4, v2
  vdiv.vv  v10, v1, v2
  vdiv.vv  v11, v4, v2
  vdiv.vv  v12, v1, v2
  vdiv.vv  v13, v4, v2
  vdiv.vv  v14, v1, v2
  vdiv.vv  v15, v4, v2
  addi     t6, t6, -1
  bnez     t6, 1b
  ROI_END
  END
