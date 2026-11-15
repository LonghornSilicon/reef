# 16 independent lw, 4 bytes apart.
# Generated for the LSU study (docs/lsu-study.md). 16 accesses per
# iteration, unrolled, ITERS times.
.include "common.inc"
.ifndef ITERS
  .equ ITERS, 20
.endif
  START
  li    t6, ITERS
  ROI_BEGIN
1:
  la    a4, buf
  lw    a0, 0(a4)
  lw    a1, 4(a4)
  lw    a2, 8(a4)
  lw    a3, 12(a4)
  lw    a5, 16(a4)
  lw    a6, 20(a4)
  lw    a7, 24(a4)
  lw    s2, 28(a4)
  lw    a0, 32(a4)
  lw    a1, 36(a4)
  lw    a2, 40(a4)
  lw    a3, 44(a4)
  lw    a5, 48(a4)
  lw    a6, 52(a4)
  lw    a7, 56(a4)
  lw    s2, 60(a4)
  addi  t6, t6, -1
  bnez  t6, 1b
  add   t1, a0, s2
  ROI_END
  END

  .data
  .balign 16

buf: .space 64
