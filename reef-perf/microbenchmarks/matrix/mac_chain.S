# 200 dependent vtmmu into the same accumulator (mt0-mt3): each waits for the
# previous one's write (hitRaw, zvt_pe_array.sv:196-207). Measures MAC
# latency. Operands v8..v22 are never loaded: tile contents are not timed.
.include "common.inc"
.include "matrix/zvt.inc"
  START
  MS_ON
  li    a0, MTYPE_I8_I32
  li    a1, VTYPE_E8
  MSETMTYPE a0, a1
  li    a0, 16
  MSETTN x0, a0
  VTZERO x0
  li    t6, 200
  ROI_BEGIN
  .balign 16
1:
  VTMMU x0, x8, x16     # mt0 += v8.. x v16..
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
