# 16 independent lw, 16 bytes apart.
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
  lw    a1, 16(a4)
  lw    a2, 32(a4)
  lw    a3, 48(a4)
  lw    a5, 64(a4)
  lw    a6, 80(a4)
  lw    a7, 96(a4)
  lw    s2, 112(a4)
  lw    a0, 128(a4)
  lw    a1, 144(a4)
  lw    a2, 160(a4)
  lw    a3, 176(a4)
  lw    a5, 192(a4)
  lw    a6, 208(a4)
  lw    a7, 224(a4)
  lw    s2, 240(a4)
  addi  t6, t6, -1
  bnez  t6, 1b
  add   t1, a0, s2
  ROI_END
  END

  .data
  .balign 16

buf: .space 256
