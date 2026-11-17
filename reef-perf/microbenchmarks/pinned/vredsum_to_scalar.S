# vredsum.vs, vmv.x.s and a scalar add of the result into the next
# reduction's start value: the reduction and the vector-to-scalar path in one
# chain.
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
  li    t6, ITERS
  la    a4, buf
  ROI_BEGIN
  .balign 16
1:
  vredsum.vs v10, v8, v9
  vmv.x.s a0, v10
  vmv.s.x v9, a0
  vredsum.vs v10, v8, v9
  vmv.x.s a0, v10
  vmv.s.x v9, a0
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
  .data
  .balign 16
buf: .space 512
