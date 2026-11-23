# One vle8.v + 15 independent scalar instructions and a branch, ITERS times.
# With a 16-entry retirement buffer the loop runs at the load's dispatch-to-
# retire time.
.include "common.inc"
.ifndef ITERS
  .equ ITERS, 50
.endif
  START
  li    t0, 16
  vsetvli x0, t0, e8, m1, ta, ma
  li    t6, ITERS
  la    a4, buf
  ROI_BEGIN
  .balign 16
1:
  vle8.v v8, (a4)
  addi  s9, s9, 1
  addi  s0, s0, 1
  addi  s1, s1, 1
  addi  a1, a1, 1
  addi  a2, a2, 1
  addi  a3, a3, 1
  addi  a6, a6, 1
  addi  a7, a7, 1
  addi  t0, t0, 1
  addi  t1, t1, 1
  addi  t2, t2, 1
  addi  t3, t3, 1
  addi  t4, t4, 1
  addi  t5, t5, 1
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
  .data
  .balign 16
buf: .space 64
