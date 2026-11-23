# A loop of 40 independent addi, 10 fetch lines: longer than the 8-entry
# instruction buffer, so fetch runs ahead of dispatch all the way round.
.include "common.inc"
.ifndef ITERS
  .equ ITERS, 50
.endif
  START
  li    t6, ITERS
  la    a4, buf
  ROI_BEGIN
  .balign 16
1:
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
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
  .data
  .balign 16
buf: .space 512
