# 16 vle32.v per iteration at e32 m1 (4 elements).
.include "common.inc"
.ifndef ITERS
  .equ ITERS, 50
.endif
  START
  li    t0, 4
  vsetvli x0, t0, e32, m1, ta, ma
  li    t6, ITERS
  la    a4, buf
  ROI_BEGIN
  .balign 16
1:
  vle32.v v8, (a4)
  vle32.v v8, (a4)
  vle32.v v8, (a4)
  vle32.v v8, (a4)
  vle32.v v8, (a4)
  vle32.v v8, (a4)
  vle32.v v8, (a4)
  vle32.v v8, (a4)
  vle32.v v8, (a4)
  vle32.v v8, (a4)
  vle32.v v8, (a4)
  vle32.v v8, (a4)
  vle32.v v8, (a4)
  vle32.v v8, (a4)
  vle32.v v8, (a4)
  vle32.v v8, (a4)
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
  .data
  .balign 16
buf: .space 512
