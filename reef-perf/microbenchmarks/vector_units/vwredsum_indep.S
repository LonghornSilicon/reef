# Independent stream: 8 x vwredsum.vs per iteration, 250 iterations.
# Measures the vector widening reduction throughput.
.include "common.inc"
  START
  vsetivli zero, 8, e16, m1, ta, ma
  vmv.v.i  v1, 1
  vmv.v.i  v2, 1
  li       t6, 250
  ROI_BEGIN
  .balign 16            # loop at slot 0 of a 16-byte fetch block
1:
  vwredsum.vs v8, v2, v1
  vwredsum.vs v9, v2, v1
  vwredsum.vs v10, v2, v1
  vwredsum.vs v11, v2, v1
  vwredsum.vs v12, v2, v1
  vwredsum.vs v13, v2, v1
  vwredsum.vs v14, v2, v1
  vwredsum.vs v15, v2, v1
  addi     t6, t6, -1
  bnez     t6, 1b
  ROI_END
  END
