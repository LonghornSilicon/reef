# ITERS (50) x 4 independent fp32 vtfmm.tvv over four accumulators.
# Held out: written after the model was calibrated; nothing is fitted to it.
.include "common.inc"
.include "matrix/zvt.inc"
.ifndef ITERS
  .equ ITERS, 50         # loop count; override with --defsym ITERS=<n>
.endif
  START
  MS_ON
  li    a0, MTYPE_E32
  li    a1, VTYPE_E32
  MSETMTYPE a0, a1
  li    a0, 16
  MSETTN x0, a0
  VTZERO x0
  VTZERO x8
  VTZERO x16
  VTZERO x24
  li    a0, MTYPE_F32_F32
  li    a1, VTYPE_E32
  MSETMTYPE a0, a1
  li    a0, 16
  MSETTN x0, a0
  li    t6, ITERS
  ROI_BEGIN
  .balign 16
1:
  VTFMM x0, x8, x16
  VTFMM x8, x8, x16
  VTFMM x16, x8, x16
  VTFMM x24, x8, x16
  addi  t6, t6, -1
  bnez  t6, 1b
  # Read the last result back so ROI_END waits for it.
  li    a0, MTYPE_E32
  li    a1, VTYPE_E32
  MSETMTYPE a0, a1
  li    a0, 16
  MSETTN x0, a0
  li    a0, (12 << 27)  # TSS: tile 12, row 0
  VTMV_VT x4, a0
  vmv.x.s t0, v4
  add   t1, t0, t0
  ROI_END
  END
