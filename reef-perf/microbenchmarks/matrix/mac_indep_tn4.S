# ITERS (50) x 4 independent int8 vtmmu, tm = 16, tn = 4, tk = 4: fewer
# columns. The RTL's busy count has no tn term (case 3), which a chain hides.
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
  li    a0, (16 << 10) | (4 << 5) | 3
  li    a1, VTYPE_E8
  MSETMTYPE a0, a1
  li    a0, 4
  MSETTN x0, a0
  li    t6, ITERS
  ROI_BEGIN
  .balign 16
1:
  VTMMU x0, x8, x16
  VTMMU x8, x8, x16
  VTMMU x16, x8, x16
  VTMMU x24, x8, x16
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
