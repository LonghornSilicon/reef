# 200 vtmmu rotating over four accumulators (mt0, mt4, mt8, mt12), so no
# two in a row share a tile. Measures MAC throughput.
.include "common.inc"
.include "matrix/zvt.inc"
  START
  MS_ON
  li    a0, MTYPE_I8_I32
  li    a1, VTYPE_E8
  MSETMTYPE a0, a1
  li    a0, 16
  MSETTN x0, a0
  li    t6, 50
  ROI_BEGIN
  .balign 16
1:
  VTMMU x0, x8, x16     # mt0
  VTMMU x8, x8, x16     # mt4
  VTMMU x16, x8, x16    # mt8
  VTMMU x24, x8, x16    # mt12
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
