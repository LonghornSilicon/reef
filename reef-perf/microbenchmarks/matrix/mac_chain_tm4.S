# ITERS (200) dependent int8 vtmmu into mt0-mt3 with tm = 4, tn = 16, tk = 4:
# a partial tile (fewer rows). Tests the PE-busy count for small tm
# (docs/matrix.md, case 3) and the hitRaw window when busy < 4 (case 9).
# Held out: written after the model was calibrated; nothing is fitted to it.
.include "common.inc"
.include "matrix/zvt.inc"
.ifndef ITERS
  .equ ITERS, 200         # loop count; override with --defsym ITERS=<n>
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
  li    a0, (4 << 10) | (4 << 5) | 3
  li    a1, VTYPE_E8
  MSETMTYPE a0, a1
  li    a0, 16
  MSETTN x0, a0
  li    t6, ITERS
  ROI_BEGIN
  .balign 16
1:
  VTMMU x0, x8, x16
  addi  t6, t6, -1
  bnez  t6, 1b
  # Read the last result back so ROI_END waits for it.
  li    a0, MTYPE_E32
  li    a1, VTYPE_E32
  MSETMTYPE a0, a1
  li    a0, 16
  MSETTN x0, a0
  li    a0, (0 << 27)  # TSS: tile 0, row 0
  VTMV_VT x4, a0
  vmv.x.s t0, v4
  add   t1, t0, t0
  ROI_END
  END
