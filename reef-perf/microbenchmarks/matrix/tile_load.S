# 16 rows of a 16x16 int8 tile loaded with vtle8, 20 times (320 loads).
# Measures tile-load throughput through the LSU.
.include "common.inc"
.include "matrix/zvt.inc"
  START
  MS_ON
  li    a0, MTYPE_E8
  li    a1, VTYPE_E8
  MSETMTYPE a0, a1
  li    a0, 16
  MSETTN x0, a0
  li    t6, 20
  ROI_BEGIN
1:
  la    a4, buf
  li    a5, 0           # TSS: tile 0, row 0
  li    t5, 16
2:
  VTLE8 a4, a5
  addi  a4, a4, 16
  addi  a5, a5, 1
  addi  t5, t5, -1
  bnez  t5, 2b
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END

  .data
  .balign 16
buf: .space 256
