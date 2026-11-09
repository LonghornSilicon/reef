# ITERS (50) times: one vtmmu, then vtdiscard. On CoralNPU vtdiscard
# dispatches only when the retirement buffer is empty, the RVV backend
# (engine included) is idle and the LSU is idle (Decode.scala:601-607), and
# alone from slot 0 (:197); the engine itself never sees it (the RVV decoder
# retires it). Measures what that interlock costs per MAC (B5). Held out.
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
  li    a0, MTYPE_I8_I32
  li    a1, VTYPE_E8
  MSETMTYPE a0, a1
  li    a0, 16
  MSETTN x0, a0
  li    t6, ITERS
  ROI_BEGIN
  .balign 16
1:
  VTMMU x0, x8, x16     # mt0 += ...
  VTDISCARD
  addi  t6, t6, -1
  bnez  t6, 1b
  # Read a tile back so ROI_END waits for the engine.
  li    a0, MTYPE_E32
  li    a1, VTYPE_E32
  MSETMTYPE a0, a1
  li    a0, 16
  MSETTN x0, a0
  li    a0, (0 << 27)   # TSS: tile 0, row 0
  VTMV_VT x4, a0
  vmv.x.s t0, v4
  add   t1, t0, t0
  ROI_END
  END
