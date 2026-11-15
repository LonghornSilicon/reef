# Dependent chain: 8 x vfdiv.vv per iteration, 250 iterations.
# Measures the vector FP divide latency (x / -1.0).
.include "common.inc"
  START
  vsetivli zero, 4, e32, m1, ta, ma
  li       t0, 0x3f800000   # 1.0f
  vmv.v.x  v1, t0
  li       t0, 0x40000000   # 2.0f
  vmv.v.x  v4, t0
  li       t0, 0xbf800000   # -1.0f
  vmv.v.x  v2, t0
  li       t6, 250
  ROI_BEGIN
  .balign 16            # loop at slot 0 of a 16-byte fetch block
1:
  vfdiv.vv v1, v1, v2
  vfdiv.vv v1, v1, v2
  vfdiv.vv v1, v1, v2
  vfdiv.vv v1, v1, v2
  vfdiv.vv v1, v1, v2
  vfdiv.vv v1, v1, v2
  vfdiv.vv v1, v1, v2
  vfdiv.vv v1, v1, v2
  addi     t6, t6, -1
  bnez     t6, 1b
  ROI_END
  END
