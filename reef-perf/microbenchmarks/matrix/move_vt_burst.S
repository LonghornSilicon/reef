# 16 vtmv.v.t per loop, 32-bit tiles (one row of mt0-mt3 into a 4-register
# group, rotating v4, v8, v12, v16 so no two are dependent; 4 uops each),
# unrolled, ITERS (20) times. Move-to-vector throughput: each uop waits
# for the RVV backend to take it (isMv2Rvv, zvt_ctrl.sv:605-607). C4.
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
  li    a5, 0           # TSS: tile 0, row 0
  VTMV_VT x4, a5
  addi  a5, a5, 1
  VTMV_VT x8, a5
  addi  a5, a5, 1
  VTMV_VT x12, a5
  addi  a5, a5, 1
  VTMV_VT x16, a5
  addi  a5, a5, 1
  VTMV_VT x4, a5
  addi  a5, a5, 1
  VTMV_VT x8, a5
  addi  a5, a5, 1
  VTMV_VT x12, a5
  addi  a5, a5, 1
  VTMV_VT x16, a5
  addi  a5, a5, 1
  VTMV_VT x4, a5
  addi  a5, a5, 1
  VTMV_VT x8, a5
  addi  a5, a5, 1
  VTMV_VT x12, a5
  addi  a5, a5, 1
  VTMV_VT x16, a5
  addi  a5, a5, 1
  VTMV_VT x4, a5
  addi  a5, a5, 1
  VTMV_VT x8, a5
  addi  a5, a5, 1
  VTMV_VT x12, a5
  addi  a5, a5, 1
  VTMV_VT x16, a5
  addi  a5, a5, 1
  addi  t6, t6, -1
  bnez  t6, 1b
  vmv.x.s t0, v16       # ROI_END waits for the last move
  add   t1, t0, t0
  ROI_END
  END
