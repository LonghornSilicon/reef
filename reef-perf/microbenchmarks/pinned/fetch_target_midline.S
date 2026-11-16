# A taken loop branch whose target is the third slot of a fetch line, and a
# forward jal to the second slot of another: lines that are only partly used.
.include "common.inc"
.ifndef ITERS
  .equ ITERS, 50
.endif
  START
  li    t6, ITERS
  la    a4, buf
  ROI_BEGIN
  .balign 16
  nop
  nop
1:
  addi  a0, a0, 1
  jal   x0, 2f
  nop
  nop
  nop
  nop
  nop
2:
  addi  a1, a1, 1
  addi  a2, a2, 1
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
  .data
  .balign 16
buf: .space 512
