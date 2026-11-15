# 8 pairs per loop of vtle32 into a row of mt0-mt3 and vtse32 of the same
# row, unrolled, ITERS (20) times. Each store reads what its load just
# wrote: mtx_tile_write_latency + mtx_tile_read_latency, through the
# lsu2vme and vme2lsu FIFOs. C4.
.include "common.inc"
.include "matrix/zvt.inc"
.ifndef ITERS
  .equ ITERS, 20         # loop count; override with --defsym ITERS=<n>
.endif
  START
  MS_ON
  li    s3, MTYPE_E32
  li    s4, VTYPE_E32
  li    s5, MTYPE_I8_I32
  li    s6, VTYPE_E8
  li    s7, 16
  MSETMTYPE s3, s4
  MSETTN x0, s7
  li    t6, ITERS
  ROI_BEGIN
1:
  la    a4, buf
  la    a6, buf2
  li    a5, 0           # TSS: tile 0, row 0
  VTLE32 a4, a5
  VTSE32 a6, a5
  addi  a4, a4, 64
  addi  a6, a6, 64
  addi  a5, a5, 1
  VTLE32 a4, a5
  VTSE32 a6, a5
  addi  a4, a4, 64
  addi  a6, a6, 64
  addi  a5, a5, 1
  VTLE32 a4, a5
  VTSE32 a6, a5
  addi  a4, a4, 64
  addi  a6, a6, 64
  addi  a5, a5, 1
  VTLE32 a4, a5
  VTSE32 a6, a5
  addi  a4, a4, 64
  addi  a6, a6, 64
  addi  a5, a5, 1
  VTLE32 a4, a5
  VTSE32 a6, a5
  addi  a4, a4, 64
  addi  a6, a6, 64
  addi  a5, a5, 1
  VTLE32 a4, a5
  VTSE32 a6, a5
  addi  a4, a4, 64
  addi  a6, a6, 64
  addi  a5, a5, 1
  VTLE32 a4, a5
  VTSE32 a6, a5
  addi  a4, a4, 64
  addi  a6, a6, 64
  addi  a5, a5, 1
  VTLE32 a4, a5
  VTSE32 a6, a5
  addi  a4, a4, 64
  addi  a6, a6, 64
  addi  a5, a5, 1
  addi  t6, t6, -1
  bnez  t6, 1b
  # Read mt0-mt3 row 0 back so ROI_END waits for the engine.
  MSETMTYPE s3, s4
  MSETTN x0, s7
  li    a0, 0
  VTMV_VT x4, a0
  vmv.x.s t0, v4
  add   t1, t0, t0
  ROI_END
  END

  .data
  .balign 16
buf: .space 1024
buf2: .space 1024
