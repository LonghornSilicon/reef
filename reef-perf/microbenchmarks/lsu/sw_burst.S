# 16 independent sw, 4 bytes apart.
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
  sw    zero, 0(a4)
  sw    zero, 4(a4)
  sw    zero, 8(a4)
  sw    zero, 12(a4)
  sw    zero, 16(a4)
  sw    zero, 20(a4)
  sw    zero, 24(a4)
  sw    zero, 28(a4)
  sw    zero, 32(a4)
  sw    zero, 36(a4)
  sw    zero, 40(a4)
  sw    zero, 44(a4)
  sw    zero, 48(a4)
  sw    zero, 52(a4)
  sw    zero, 56(a4)
  sw    zero, 60(a4)
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END

  .data
  .balign 16

buf: .space 64
