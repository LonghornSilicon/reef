# 8 dependent masked vadd.vv (v0.t) per iteration.
.include "common.inc"
.ifndef ITERS
  .equ ITERS, 50
.endif
  START
  li    t0, 4
  vsetvli x0, t0, e32, m1, ta, ma
  li    t0, 5
  vmv.v.x v8, t0
  vmv.v.x v9, t0
  vmset.m v0
  li    t6, ITERS
  la    a4, buf
  ROI_BEGIN
  .balign 16
1:
  vadd.vv v8, v8, v9, v0.t
  vadd.vv v8, v8, v9, v0.t
  vadd.vv v8, v8, v9, v0.t
  vadd.vv v8, v8, v9, v0.t
  vadd.vv v8, v8, v9, v0.t
  vadd.vv v8, v8, v9, v0.t
  vadd.vv v8, v8, v9, v0.t
  vadd.vv v8, v8, v9, v0.t
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
  .data
  .balign 16
buf: .space 512
