# Independent stream: 8 x vfadd.vv per iteration, 250 iterations.
# Measures the vector FP add throughput.
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
  vfadd.vv v8, v1, v3
  vfadd.vv v9, v1, v3
  vfadd.vv v10, v1, v3
  vfadd.vv v11, v1, v3
  vfadd.vv v12, v1, v3
  vfadd.vv v13, v1, v3
  vfadd.vv v14, v1, v3
  vfadd.vv v15, v1, v3
  addi     t6, t6, -1
  bnez     t6, 1b
  ROI_END
  END
