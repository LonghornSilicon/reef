# Independent stream: 6 x vredsum.vs per iteration, 250 iterations.
# Measures the vector reduction throughput, 4 uops per instruction; one vector ROB entry.
.include "common.inc"
  START
  vsetivli zero, 16, e32, m4, ta, ma
  vmv.v.i  v0, 1
  vmv.v.i  v4, 1
  li       t6, 250
  ROI_BEGIN
  .balign 16            # loop at slot 0 of a 16-byte fetch block
1:
  vredsum.vs v8, v4, v0
  vredsum.vs v12, v4, v0
  vredsum.vs v16, v4, v0
  vredsum.vs v20, v4, v0
  vredsum.vs v24, v4, v0
  vredsum.vs v28, v4, v0
  addi     t6, t6, -1
  bnez     t6, 1b
  ROI_END
  END
