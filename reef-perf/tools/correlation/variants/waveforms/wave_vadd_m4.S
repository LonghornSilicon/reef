# vadd_m4_indep waveform variant: 4 iterations of 6 independent vadd.vv at
# LMUL = 4 (4 uops each).
.include "common.inc"
  START
  vsetivli zero, 16, e32, m4, ta, ma
  vmv.v.i  v0, 1
  vmv.v.i  v4, 1
  li       t6, 4
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
