# 4 independent vtmmu (mt0-3, mt4-7, mt8-11, mt12-15) + 8 independent addi,
# ITERS (50) times. How far dispatch runs ahead of the engine: the Zvt
# reservation station holds 8 uops (ZVT_RS_DEPTH) = 2 vtmmu, the model's
# queue 4 instructions (mtx_queue_entries). C4.
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
  MSETMTYPE s5, s6
  MSETTN x0, s7
  li    t6, ITERS
  ROI_BEGIN
  .balign 16
1:
  VTMMU x0, x8, x16
  VTMMU x8, x8, x16
  VTMMU x16, x8, x16
  VTMMU x24, x8, x16
  addi  s9, s9, 1
  addi  s0, s0, 1
  addi  s1, s1, 1
  addi  a1, a1, 1
  addi  a2, a2, 1
  addi  a3, a3, 1
  addi  a6, a6, 1
  addi  a7, a7, 1
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
