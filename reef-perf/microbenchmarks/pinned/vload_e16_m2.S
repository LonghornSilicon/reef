# 16 vle16.v per iteration at e16 m2 (16 elements, 2 registers).
.include "common.inc"
.ifndef ITERS
  .equ ITERS, 50
.endif
  START
  li    t0, 16
  vsetvli x0, t0, e16, m2, ta, ma
  li    t6, ITERS
  la    a4, buf
  ROI_BEGIN
  .balign 16
1:
  vle16.v v8, (a4)
  vle16.v v8, (a4)
  vle16.v v8, (a4)
  vle16.v v8, (a4)
  vle16.v v8, (a4)
  vle16.v v8, (a4)
  vle16.v v8, (a4)
  vle16.v v8, (a4)
  vle16.v v8, (a4)
  vle16.v v8, (a4)
  vle16.v v8, (a4)
  vle16.v v8, (a4)
  vle16.v v8, (a4)
  vle16.v v8, (a4)
  vle16.v v8, (a4)
  vle16.v v8, (a4)
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
  .data
  .balign 16
buf: .space 512
