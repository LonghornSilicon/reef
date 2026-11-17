# 16 vle8.v per iteration with vl 3 at e8 m4: a register group of which only
# the first register has elements.
.include "common.inc"
.ifndef ITERS
  .equ ITERS, 50
.endif
  START
  li    t0, 3
  vsetvli x0, t0, e8, m4, ta, ma
  li    t6, ITERS
  la    a4, buf
  ROI_BEGIN
  .balign 16
1:
  vle8.v v8, (a4)
  vle8.v v8, (a4)
  vle8.v v8, (a4)
  vle8.v v8, (a4)
  vle8.v v8, (a4)
  vle8.v v8, (a4)
  vle8.v v8, (a4)
  vle8.v v8, (a4)
  vle8.v v8, (a4)
  vle8.v v8, (a4)
  vle8.v v8, (a4)
  vle8.v v8, (a4)
  vle8.v v8, (a4)
  vle8.v v8, (a4)
  vle8.v v8, (a4)
  vle8.v v8, (a4)
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
  .data
  .balign 16
buf: .space 512
