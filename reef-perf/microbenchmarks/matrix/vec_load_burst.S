# 16 vle8.v at LMUL 1 (one 16-byte register each), unrolled, ITERS (20)
# times. The vector reference for tile_load_burst: the same
# shape with RVV instead of Zvt, so the two LSU paths can be compared on
# one core (docs/lsu-tile-beats.md). No matrix instructions.
.include "common.inc"
.ifndef ITERS
  .equ ITERS, 20         # loop count; override with --defsym ITERS=<n>
.endif
  START
  li    t0, 16
  vsetvli x0, t0, e8, m1, ta, ma
  li    t6, ITERS
  ROI_BEGIN
1:
  la    a4, buf
  vle8.v v8, (a4)
  addi  a4, a4, 16
  vle8.v v9, (a4)
  addi  a4, a4, 16
  vle8.v v10, (a4)
  addi  a4, a4, 16
  vle8.v v11, (a4)
  addi  a4, a4, 16
  vle8.v v12, (a4)
  addi  a4, a4, 16
  vle8.v v13, (a4)
  addi  a4, a4, 16
  vle8.v v14, (a4)
  addi  a4, a4, 16
  vle8.v v15, (a4)
  addi  a4, a4, 16
  vle8.v v16, (a4)
  addi  a4, a4, 16
  vle8.v v17, (a4)
  addi  a4, a4, 16
  vle8.v v18, (a4)
  addi  a4, a4, 16
  vle8.v v19, (a4)
  addi  a4, a4, 16
  vle8.v v20, (a4)
  addi  a4, a4, 16
  vle8.v v21, (a4)
  addi  a4, a4, 16
  vle8.v v22, (a4)
  addi  a4, a4, 16
  vle8.v v23, (a4)
  addi  a4, a4, 16
  addi  t6, t6, -1
  bnez  t6, 1b
  # Read the last register back so ROI_END waits for it.
  vmv.x.s t0, v23
  add   t1, t0, t0
  ROI_END
  END

  .data
  .balign 16
buf: .space 256
