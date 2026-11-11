# vtmv.t.v into mt0-mt3 (row 0 from v4-v7), then a vtmmu accumulating
# into mt0-mt3, ITERS (50) times. The MAC waits for the move's tile write
# (mtx_move_latency); the next move waits for the MAC to drain. C4.
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
  MSETMTYPE s3, s4
  MSETTN x0, s7
  VTMV_TV a5, x4
  MSETMTYPE s5, s6
  MSETTN x0, s7
  VTMMU x0, x8, x16     # mt0 += v8.. x v16..
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
