# Pointer chase through lw: each address depends on the previous load.
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
  lw    t0, 0(a4)
  add   a4, a4, t0
  lw    t0, 0(a4)
  add   a4, a4, t0
  lw    t0, 0(a4)
  add   a4, a4, t0
  lw    t0, 0(a4)
  add   a4, a4, t0
  lw    t0, 0(a4)
  add   a4, a4, t0
  lw    t0, 0(a4)
  add   a4, a4, t0
  lw    t0, 0(a4)
  add   a4, a4, t0
  lw    t0, 0(a4)
  add   a4, a4, t0
  lw    t0, 0(a4)
  add   a4, a4, t0
  lw    t0, 0(a4)
  add   a4, a4, t0
  lw    t0, 0(a4)
  add   a4, a4, t0
  lw    t0, 0(a4)
  add   a4, a4, t0
  lw    t0, 0(a4)
  add   a4, a4, t0
  lw    t0, 0(a4)
  add   a4, a4, t0
  lw    t0, 0(a4)
  add   a4, a4, t0
  lw    t0, 0(a4)
  add   a4, a4, t0
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END

  .data
  .balign 16

buf: .space 16
