# Dependent chain: 8 x vrgather.vv per iteration, 250 iterations.
# Measures the vector gather latency (vd may not overlap a source, so two registers alternate).
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
  vrgather.vv v4, v1, v3
  vrgather.vv v1, v4, v3
  vrgather.vv v4, v1, v3
  vrgather.vv v1, v4, v3
  vrgather.vv v4, v1, v3
  vrgather.vv v1, v4, v3
  vrgather.vv v4, v1, v3
  vrgather.vv v1, v4, v3
  addi     t6, t6, -1
  bnez     t6, 1b
  ROI_END
  END
