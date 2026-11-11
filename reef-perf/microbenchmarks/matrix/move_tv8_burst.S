# 16 vtmv.t.v per loop, 8-bit tiles (one row of mt0 from v8, 1 uop
# each), unrolled, ITERS (20) times. With move_tv_burst, separates the
# per-instruction from the per-uop move cost. C4.
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
  li    s8, MTYPE_E8
  MSETMTYPE s8, s6
  MSETTN x0, s7
  li    t6, ITERS
  ROI_BEGIN
1:
  li    a5, 0           # TSS: tile 0, row 0
  VTMV_TV a5, x8
  addi  a5, a5, 1
  VTMV_TV a5, x8
  addi  a5, a5, 1
  VTMV_TV a5, x8
  addi  a5, a5, 1
  VTMV_TV a5, x8
  addi  a5, a5, 1
  VTMV_TV a5, x8
  addi  a5, a5, 1
  VTMV_TV a5, x8
  addi  a5, a5, 1
  VTMV_TV a5, x8
  addi  a5, a5, 1
  VTMV_TV a5, x8
  addi  a5, a5, 1
  VTMV_TV a5, x8
  addi  a5, a5, 1
  VTMV_TV a5, x8
  addi  a5, a5, 1
  VTMV_TV a5, x8
  addi  a5, a5, 1
  VTMV_TV a5, x8
  addi  a5, a5, 1
  VTMV_TV a5, x8
  addi  a5, a5, 1
  VTMV_TV a5, x8
  addi  a5, a5, 1
  VTMV_TV a5, x8
  addi  a5, a5, 1
  VTMV_TV a5, x8
  addi  a5, a5, 1
  addi  t6, t6, -1
  bnez  t6, 1b
  # Read mt0 row 0 back so ROI_END waits for the engine.
  li    a0, 0
  VTMV_VT x4, a0
  vmv.x.s t0, v4
  add   t1, t0, t0
  ROI_END
  END
