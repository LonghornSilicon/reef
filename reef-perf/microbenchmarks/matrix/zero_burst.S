# 16 vtzero per loop (32-bit tiles: each zeroes a group of 4 tiles),
# rotating over mt0, mt4, mt8, mt12, unrolled, ITERS (20) times.
# Measures how long vtzero holds the controller (PROCESS_DELAY,
# zvt_ctrl.sv:435-445, :613: mtx_zero_cycles). C4.
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
  VTZERO x0
  VTZERO x8
  VTZERO x16
  VTZERO x24
  VTZERO x0
  VTZERO x8
  VTZERO x16
  VTZERO x24
  VTZERO x0
  VTZERO x8
  VTZERO x16
  VTZERO x24
  VTZERO x0
  VTZERO x8
  VTZERO x16
  VTZERO x24
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
