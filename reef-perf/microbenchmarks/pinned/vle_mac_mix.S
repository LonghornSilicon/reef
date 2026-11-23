# A vle8.v into each MAC operand, then the MAC, twice per iteration: the
# operand loads and the engine sharing the backend, as in a GEMM k-step.
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
  vle8.v v8, (a4)
  vle8.v v16, (a4)
  VTMMU x0, x8, x16
  vle8.v v9, (a4)
  vle8.v v17, (a4)
  VTMMU x8, x9, x17
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
  .data
  .balign 16
buf: .space 512
