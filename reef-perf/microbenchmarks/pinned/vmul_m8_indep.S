# 4 independent vmul.vv per iteration at e32 m8 (8 uops each): more uops in
# flight than the 16-entry vector ROB holds.
.include "common.inc"
.ifndef ITERS
  .equ ITERS, 50
.endif
  START
  li    t0, 32
  vsetvli x0, t0, e32, m8, ta, ma
  li    t0, 5
  vmv.v.x v8, t0
  li    t6, ITERS
  la    a4, buf
  ROI_BEGIN
  .balign 16
1:
  vmul.vv v16, v8, v8
  vmul.vv v24, v8, v8
  vmul.vv v16, v8, v8
  vmul.vv v24, v8, v8
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
  .data
  .balign 16
buf: .space 512
