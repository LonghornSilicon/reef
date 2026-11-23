# A vsext.vf2 into a vwadd-free chain: vsext.vf4 of a register, then vadd.vv
# of the result into an accumulator, 4 pairs per iteration.
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
  vmv.v.i v12, 0
  li    t6, ITERS
  la    a4, buf
  ROI_BEGIN
  .balign 16
1:
  vsext.vf4 v10, v8
  vadd.vv v12, v12, v10
  vsext.vf4 v10, v8
  vadd.vv v12, v12, v10
  vsext.vf4 v10, v8
  vadd.vv v12, v12, v10
  vsext.vf4 v10, v8
  vadd.vv v12, v12, v10
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
  .data
  .balign 16
buf: .space 512
