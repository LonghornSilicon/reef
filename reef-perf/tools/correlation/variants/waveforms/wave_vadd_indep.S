# vadd_indep waveform variant: 6 iterations of 8 independent vadd.vv at
# LMUL = 1. Shows the fetch bubble of roadmap B21 and the branch rule of B07.
.include "common.inc"
  START
  vsetivli zero, 4, e32, m1, ta, ma
  vmv.v.i  v1, 1
  vmv.v.i  v2, 1
  vmv.v.i  v3, 0
  li       t6, 6
  ROI_BEGIN
  .balign 16            # loop at slot 0 of a 16-byte fetch block
1:
  vadd.vv  v8, v1, v2
  vadd.vv  v9, v1, v2
  vadd.vv  v10, v1, v2
  vadd.vv  v11, v1, v2
  vadd.vv  v12, v1, v2
  vadd.vv  v13, v1, v2
  vadd.vv  v14, v1, v2
  vadd.vv  v15, v1, v2
  addi     t6, t6, -1
  bnez     t6, 1b
  ROI_END
  END
