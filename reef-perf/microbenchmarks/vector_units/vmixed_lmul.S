# Mixed LMUL: 1 x vadd.vv at LMUL=8 (8 uops), then 4 x vadd.vv at LMUL=1, with
# a vsetvli before each part, per iteration, 250 iterations (roadmap B16).
.include "common.inc"
  START
  li       t0, 32
  vsetvli  zero, t0, e32, m8, ta, ma
  vmv.v.i  v8, 1
  vmv.v.i  v16, 1
  li       t6, 250
  ROI_BEGIN
  .balign 16            # loop at slot 0 of a 16-byte fetch block
1:
  vsetvli  zero, t0, e32, m8, ta, ma
  vadd.vv  v24, v8, v16
  vsetivli zero, 4, e32, m1, ta, ma
  vadd.vv  v1, v8, v16
  vadd.vv  v2, v8, v16
  vadd.vv  v3, v8, v16
  vadd.vv  v4, v8, v16
  addi     t6, t6, -1
  bnez     t6, 1b
  ROI_END
  END
