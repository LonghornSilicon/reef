# ITERS (100) times: two vtmmu with a misaligned operand, then one with odd
# but aligned operands. For int8 the RVV decoder requires vs1/vs2 register
# bits 2:1 = 0 (check_vme_vs{1,2}_align, rvv_backend_decode_unit_ari.sv
# :3219-3233); otherwise the vtmm retires as a no-op, with no trap and no
# tile write (rvv_backend_decode.sv:54-62). v10 and v18 fail the check; v9
# and v17 pass it. Measures what a rejected vtmm costs (B2).
.include "common.inc"
.include "matrix/zvt.inc"
.ifndef ITERS
  .equ ITERS, 100        # loop count; override with --defsym ITERS=<n>
.endif
  START
  MS_ON
  li    a0, MTYPE_E32
  li    a1, VTYPE_E32
  MSETMTYPE a0, a1
  li    a0, 16
  MSETTN x0, a0
  VTZERO x8              # mt4-mt7
  li    a0, MTYPE_I8_I32
  li    a1, VTYPE_E8
  MSETMTYPE a0, a1
  li    a0, 16
  MSETTN x0, a0
  li    t6, ITERS
  ROI_BEGIN
  .balign 16
1:
  VTMMU x0, x10, x16    # vs2 = v10 (bits 2:1 = 01): no-op
  VTMMU x0, x8, x18     # vs1 = v18 (bits 2:1 = 01): no-op
  VTMMU x8, x9, x17     # vs2 = v9, vs1 = v17 (bits 2:1 = 00): mt4 += ...
  addi  t6, t6, -1
  bnez  t6, 1b
  # Read the last result back so ROI_END waits for it.
  li    a0, MTYPE_E32
  li    a1, VTYPE_E32
  MSETMTYPE a0, a1
  li    a0, 16
  MSETTN x0, a0
  li    a0, (4 << 27)   # TSS: tile 4, row 0
  VTMV_VT x4, a0
  vmv.x.s t0, v4
  add   t1, t0, t0
  ROI_END
  END
