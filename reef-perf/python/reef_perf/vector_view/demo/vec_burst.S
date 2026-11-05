# Demo program for the vector view: bursts of independent LMUL=4 adds and
# multiplies. Each holds a lane for 4 cycles, so fourteen in a row outrun the
# two lanes and fill the vector command queue; pauses between bursts let it
# drain.
.include "common.inc"
  START
  li    t6, 12
  ROI_BEGIN
1:
  li    a3, 16
  vsetvli t0, a3, e32, m4, ta, ma
  .rept 2
  vadd.vv v4, v0, v0
  vmul.vv v8, v0, v0
  vadd.vv v12, v0, v0
  vmul.vv v16, v0, v0
  vadd.vv v20, v0, v0
  vmul.vv v24, v0, v0
  vadd.vv v28, v0, v0
  .endr
  .rept 24
  addi  a0, a0, 1
  .endr
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
