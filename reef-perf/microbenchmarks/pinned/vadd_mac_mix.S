# Two independent MACs and four independent vadd.vv per iteration,
# interleaved: Zvt and vector uops through the RVV backend's one dispatch.
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
  li    t6, ITERS
  la    a4, buf
  ROI_BEGIN
  .balign 16
1:
  VTMMU x0, x8, x16
  vadd.vv v2, v1, v1
  vadd.vv v3, v1, v1
  VTMMU x8, x8, x16
  vadd.vv v4, v1, v1
  vadd.vv v5, v1, v1
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
  .data
  .balign 16
buf: .space 512
