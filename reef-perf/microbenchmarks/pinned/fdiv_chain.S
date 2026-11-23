# 8 dependent fdiv.s per iteration.
.include "common.inc"
.ifndef ITERS
  .equ ITERS, 50
.endif
  START
  li    t0, 0x40400000
  fmv.w.x f1, t0
  li    t0, 0x3fc00000
  fmv.w.x f2, t0
  fmv.w.x f3, t0
  li    t6, ITERS
  la    a4, buf
  ROI_BEGIN
  .balign 16
1:
  fdiv.s f3, f3, f2
  fdiv.s f3, f3, f2
  fdiv.s f3, f3, f2
  fdiv.s f3, f3, f2
  fdiv.s f3, f3, f2
  fdiv.s f3, f3, f2
  fdiv.s f3, f3, f2
  fdiv.s f3, f3, f2
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
  .data
  .balign 16
buf: .space 512
