# A vtmmu into mt0-mt3, then vtse32 of row 0 of mt0-mt3, ITERS (50)
# times. The store's controller half waits for the MAC to drain, then
# reads the tile (mtx_tile_read_latency into vme2lsu). C4.
.include "common.inc"
.include "matrix/zvt.inc"
.ifndef ITERS
  .equ ITERS, 50         # loop count; override with --defsym ITERS=<n>
.endif
  START
  MS_ON
  li    s3, MTYPE_E32
  li    s4, VTYPE_E32
  li    s5, MTYPE_I8_I32
  li    s6, VTYPE_E8
  li    s7, 16
  la    a4, buf
  li    a5, 0           # TSS: tile 0, row 0
  li    t6, ITERS
  ROI_BEGIN
1:
  MSETMTYPE s5, s6
  MSETTN x0, s7
  VTMMU x0, x8, x16
  MSETMTYPE s3, s4
  MSETTN x0, s7
  VTSE32 a4, a5
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
