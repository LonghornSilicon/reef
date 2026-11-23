# Scalar and vector accesses interleaved, 4 of each per iteration: lw, vle8.v,
# sw, vse8.v sharing the LSU's one slot.
.include "common.inc"
.ifndef ITERS
  .equ ITERS, 50
.endif
  START
  li    t0, 16
  vsetvli x0, t0, e8, m1, ta, ma
  li    t6, ITERS
  la    a4, buf
  addi  a5, a4, 16
  addi  a6, a4, 48
  ROI_BEGIN
  .balign 16
1:
  lw    a0, 0(a4)
  vle8.v v8, (a5)
  sw    a1, 32(a4)
  vse8.v v9, (a6)
  lw    a0, 0(a4)
  vle8.v v8, (a5)
  sw    a1, 32(a4)
  vse8.v v9, (a6)
  lw    a0, 0(a4)
  vle8.v v8, (a5)
  sw    a1, 32(a4)
  vse8.v v9, (a6)
  lw    a0, 0(a4)
  vle8.v v8, (a5)
  sw    a1, 32(a4)
  vse8.v v9, (a6)
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
  .data
  .balign 16
buf: .space 512
