# Four forward jal in a row, each predicted, each skipping one instruction:
# a taken jump in every fetch line.
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
  jal   x0, .+8
  nop
  jal   x0, .+8
  nop
  jal   x0, .+8
  nop
  jal   x0, .+8
  nop
  addi  a0, a0, 1
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
  .data
  .balign 16
buf: .space 512
