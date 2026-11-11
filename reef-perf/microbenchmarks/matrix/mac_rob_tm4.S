# One vtmmu with tm = 4, tn = 16 + 15 independent addi, ITERS (50) times.
# ROB-bound like mac_rob, so the loop runs at the MAC's dispatch-to-retire
# time for a partial tile (D4: the 128-PE RTL retires tm-4 MACs sooner).
.include "common.inc"
.include "matrix/zvt.inc"
.ifndef ITERS
  .equ ITERS, 50         # loop count; override with --defsym ITERS=<n>
.endif
  START
  MS_ON
  li    s3, MTYPE_E32
  li    s4, VTYPE_E32
  li    s5, (4 << 10) | (4 << 5) | 3
  li    s6, VTYPE_E8
  li    s7, 16
  MSETMTYPE s5, s6
  li    s8, 16
  MSETTN x0, s8
  li    a5, 0           # TSS: tile 0, row 0
  li    t6, ITERS
  ROI_BEGIN
  .balign 16
1:
  VTMMU x0, x8, x16
  addi  s9, s9, 1
  addi  s0, s0, 1
  addi  s1, s1, 1
  addi  a1, a1, 1
  addi  a2, a2, 1
  addi  a3, a3, 1
  addi  a6, a6, 1
  addi  a7, a7, 1
  addi  t0, t0, 1
  addi  t1, t1, 1
  addi  t2, t2, 1
  addi  t3, t3, 1
  addi  t4, t4, 1
  addi  t5, t5, 1
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
