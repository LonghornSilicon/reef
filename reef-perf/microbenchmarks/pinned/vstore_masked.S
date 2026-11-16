# 16 masked vse8.v (v0.t) per iteration.
.include "common.inc"
.ifndef ITERS
  .equ ITERS, 50
.endif
  START
  li    t0, 16
  vsetvli x0, t0, e8, m1, ta, ma
  vmset.m v0
  li    t6, ITERS
  la    a4, buf
  ROI_BEGIN
  .balign 16
1:
  vse8.v v8, (a4), v0.t
  vse8.v v8, (a4), v0.t
  vse8.v v8, (a4), v0.t
  vse8.v v8, (a4), v0.t
  vse8.v v8, (a4), v0.t
  vse8.v v8, (a4), v0.t
  vse8.v v8, (a4), v0.t
  vse8.v v8, (a4), v0.t
  vse8.v v8, (a4), v0.t
  vse8.v v8, (a4), v0.t
  vse8.v v8, (a4), v0.t
  vse8.v v8, (a4), v0.t
  vse8.v v8, (a4), v0.t
  vse8.v v8, (a4), v0.t
  vse8.v v8, (a4), v0.t
  vse8.v v8, (a4), v0.t
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
  .data
  .balign 16
buf: .space 512
