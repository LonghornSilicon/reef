# ITERS (50) times: one vtmmu into mt0-mt3, then 8 vtle8 rows into mt4 and a
# scalar load. vtle8's controller half waits for the MAC to drain (!peBusy,
# zvt_ctrl.sv:105-106) while its LSU half runs ahead, at most
# mtx_lsu_fifo_entries beats (lsu2vme, zvt.sv:163-206); then the LSU holds.
# Measures how the lsu2vme FIFO couples the LSU to the engine.
.include "common.inc"
.include "matrix/zvt.inc"
.ifndef ITERS
  .equ ITERS, 50         # loop count; override with --defsym ITERS=<n>
.endif
  START
  MS_ON
  li    a0, MTYPE_I8_I32
  li    a1, VTYPE_E8
  MSETMTYPE a0, a1
  li    a0, 16
  MSETTN x0, a0
  VTZERO x0
  la    a4, buf
  li    t6, ITERS
  ROI_BEGIN
  .balign 16
1:
  VTMMU x0, x8, x16     # mt0 += v8.. x v16..
  li    a5, (4 << 27)   # TSS: tile 4, row 0
  VTLE8 a4, a5
  addi  a5, a5, 1
  VTLE8 a4, a5
  addi  a5, a5, 1
  VTLE8 a4, a5
  addi  a5, a5, 1
  VTLE8 a4, a5
  addi  a5, a5, 1
  VTLE8 a4, a5
  addi  a5, a5, 1
  VTLE8 a4, a5
  addi  a5, a5, 1
  VTLE8 a4, a5
  addi  a5, a5, 1
  VTLE8 a4, a5
  lw    t0, 0(a4)
  addi  t6, t6, -1
  bnez  t6, 1b
  # Read the last loaded row back so ROI_END waits for it.
  li    a0, MTYPE_E8
  li    a1, VTYPE_E8
  MSETMTYPE a0, a1
  li    a0, 16
  MSETTN x0, a0
  li    a5, (4 << 27) + 7  # TSS: tile 4, row 7
  VTMV_VT x4, a5
  vmv.x.s t0, v4
  add   t1, t0, t0
  ROI_END
  END

  .data
  .balign 16
buf: .space 256
