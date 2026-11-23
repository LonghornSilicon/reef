# A mispredicted forward branch behind 15 addi and a slow vector divide: the
# retirement buffer is full when the branch reaches the head of the
# instruction buffer.
.include "common.inc"
.ifndef ITERS
  .equ ITERS, 50
.endif
  START
  li    t0, 4
  vsetvli x0, t0, e32, m1, ta, ma
  li    t0, 0x7fffffff
  vmv.v.x v8, t0
  li    t0, 3
  vmv.v.x v10, t0
  li    t6, ITERS
  la    a4, buf
  ROI_BEGIN
  .balign 16
1:
  vdiv.vv v9, v8, v10
  addi  a0, a0, 1
  addi  a1, a1, 1
  addi  a2, a2, 1
  addi  a3, a3, 1
  addi  a5, a5, 1
  addi  a6, a6, 1
  addi  a7, a7, 1
  addi  t0, t0, 1
  addi  t1, t1, 1
  addi  t2, t2, 1
  addi  t3, t3, 1
  addi  t4, t4, 1
  addi  t5, t5, 1
  addi  s0, s0, 1
  addi  s1, s1, 1
  beq   x0, x0, 2f
  nop
2:
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
  .data
  .balign 16
buf: .space 512
