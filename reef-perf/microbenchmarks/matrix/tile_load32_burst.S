# 16 rows of a 16x16 32-bit tile loaded with vtle32, unrolled, ITERS (20)
# times: 320 loads of 4 beats each (64 bytes). With tile_load_burst it
# separates the per-beat cost from the per-instruction cost
# (docs/lsu-tile-beats.md).
.include "common.inc"
.include "matrix/zvt.inc"
.ifndef ITERS
  .equ ITERS, 20         # loop count; override with --defsym ITERS=<n>
.endif
  START
  MS_ON
  li    a0, MTYPE_E32
  li    a1, VTYPE_E32
  MSETMTYPE a0, a1
  li    a0, 16
  MSETTN x0, a0
  li    t6, ITERS
  ROI_BEGIN
1:
  la    a4, buf
  li    a5, 0           # TSS: tile 0, row 0
  VTLE32 a4, a5
  addi  a4, a4, 64
  addi  a5, a5, 1
  VTLE32 a4, a5
  addi  a4, a4, 64
  addi  a5, a5, 1
  VTLE32 a4, a5
  addi  a4, a4, 64
  addi  a5, a5, 1
  VTLE32 a4, a5
  addi  a4, a4, 64
  addi  a5, a5, 1
  VTLE32 a4, a5
  addi  a4, a4, 64
  addi  a5, a5, 1
  VTLE32 a4, a5
  addi  a4, a4, 64
  addi  a5, a5, 1
  VTLE32 a4, a5
  addi  a4, a4, 64
  addi  a5, a5, 1
  VTLE32 a4, a5
  addi  a4, a4, 64
  addi  a5, a5, 1
  VTLE32 a4, a5
  addi  a4, a4, 64
  addi  a5, a5, 1
  VTLE32 a4, a5
  addi  a4, a4, 64
  addi  a5, a5, 1
  VTLE32 a4, a5
  addi  a4, a4, 64
  addi  a5, a5, 1
  VTLE32 a4, a5
  addi  a4, a4, 64
  addi  a5, a5, 1
  VTLE32 a4, a5
  addi  a4, a4, 64
  addi  a5, a5, 1
  VTLE32 a4, a5
  addi  a4, a4, 64
  addi  a5, a5, 1
  VTLE32 a4, a5
  addi  a4, a4, 64
  addi  a5, a5, 1
  VTLE32 a4, a5
  addi  a4, a4, 64
  addi  a5, a5, 1
  addi  t6, t6, -1
  bnez  t6, 1b
  # Read the last row back so ROI_END waits for it.
  li    a5, 15          # TSS: tile 0, row 15
  VTMV_VT x4, a5
  vmv.x.s t0, v4
  add   t1, t0, t0
  ROI_END
  END

  .data
  .balign 16
buf: .space 1024
