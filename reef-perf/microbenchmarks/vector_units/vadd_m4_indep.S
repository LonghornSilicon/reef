# Independent stream: 6 x vadd.vv per iteration, 250 iterations.
# Measures the vector ALU throughput, 4 uops per instruction.
.include "common.inc"
  START
  vsetivli zero, 16, e32, m4, ta, ma
  vmv.v.i  v0, 1
  vmv.v.i  v4, 1
  li       t6, 250
  ROI_BEGIN
  .balign 16            # loop at slot 0 of a 16-byte fetch block
1:
  vadd.vv  v8, v0, v4
  vadd.vv  v12, v0, v4
  vadd.vv  v16, v0, v4
  vadd.vv  v20, v0, v4
  vadd.vv  v24, v0, v4
  vadd.vv  v28, v0, v4
  addi     t6, t6, -1
  bnez     t6, 1b
  ROI_END
  END
