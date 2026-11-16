# A vtse8 and a vle8.v alternating, 4 pairs per iteration: tile stores and
# vector loads sharing the LSU slot.
.include "common.inc"
.include "matrix/zvt.inc"
.ifndef ITERS
  .equ ITERS, 50
.endif
  START
  MS_ON
  li    a0, MTYPE_I8_I32
  li    a1, VTYPE_E8
  MSETMTYPE a0, a1
  li    a0, 16
  MSETTN x0, a0
  li    a0, MTYPE_E8
  MSETMTYPE a0, a1
  li    a0, 16
  MSETTN x0, a0
  li    a5, 0
  li    t6, ITERS
  la    a4, buf
  ROI_BEGIN
  .balign 16
1:
  VTSE8 a4, a5
  vle8.v v8, (a4)
  VTSE8 a4, a5
  vle8.v v8, (a4)
  VTSE8 a4, a5
  vle8.v v8, (a4)
  VTSE8 a4, a5
  vle8.v v8, (a4)
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
  .data
  .balign 16
buf: .space 512
