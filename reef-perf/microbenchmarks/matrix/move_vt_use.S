# A loop-carried chain, ITERS (100) times: vtmv.v.t (32-bit, mt0-mt3 row
# 0 into v4-v7), vmv.x.s of its result, and a TSS for the next move that
# depends on that scalar. Latency from a move to a vector reader:
# mtx_move_latency plus the vector side's read rule. C4.
.include "common.inc"
.include "matrix/zvt.inc"
.ifndef ITERS
  .equ ITERS, 100         # loop count; override with --defsym ITERS=<n>
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
  li    a0, 0           # TSS: tile 0, row 0
  li    t6, ITERS
  ROI_BEGIN
1:
  VTMV_VT x4, a0
  vmv.x.s t0, v4
  andi  t0, t0, 0
  add   a0, a0, t0      # next TSS depends on the read-back
  addi  t6, t6, -1
  bnez  t6, 1b
  add   t1, a0, a0
  ROI_END
  END
