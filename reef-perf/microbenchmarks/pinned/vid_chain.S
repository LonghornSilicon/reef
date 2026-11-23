# vid.v and viota.m, whose vs1 field is a sub-opcode, 4 of each per
# iteration behind a vadd.vv chain on other registers.
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
  vmset.m v0
  vmv.v.i v12, 0
  li    t6, ITERS
  la    a4, buf
  ROI_BEGIN
  .balign 16
1:
  vid.v v10
  viota.m v11, v0
  vadd.vv v12, v12, v9
  vid.v v10
  viota.m v11, v0
  vadd.vv v12, v12, v9
  vid.v v10
  viota.m v11, v0
  vadd.vv v12, v12, v9
  vid.v v10
  viota.m v11, v0
  vadd.vv v12, v12, v9
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
  .data
  .balign 16
buf: .space 512
