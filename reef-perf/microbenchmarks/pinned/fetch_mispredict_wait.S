# A forward branch that is always taken (a mispredict) and reads the result
# of a divide: the redirect waits for the branch to be dispatched, which
# waits for the divide.
.include "common.inc"
.ifndef ITERS
  .equ ITERS, 50
.endif
  START
  li    a1, 0x7fffffff
  li    a2, 3
  li    a3, 0x2aaaaaaa
  li    t6, ITERS
  la    a4, buf
  ROI_BEGIN
  .balign 16
1:
  div   a0, a1, a2
  beq   a0, a3, 2f
  nop
  nop
2:
  addi  a5, a5, 1
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
  .data
  .balign 16
buf: .space 512
