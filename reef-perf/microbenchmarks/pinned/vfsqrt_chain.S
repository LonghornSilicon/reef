# 4 dependent vfsqrt.v per iteration.
.include "common.inc"
.ifndef ITERS
  .equ ITERS, 50
.endif
  START
  li    t0, 4
  vsetvli x0, t0, e32, m1, ta, ma
  li    t0, 0x40400000
  fmv.w.x f1, t0
  vfmv.v.f v8, f1
  li    t6, ITERS
  la    a4, buf
  ROI_BEGIN
  .balign 16
1:
  vfsqrt.v v8, v8
  vfsqrt.v v8, v8
  vfsqrt.v v8, v8
  vfsqrt.v v8, v8
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
  .data
  .balign 16
buf: .space 512
