# Dependent multiply chain (500 x 8 muls). Measures MLU latency.
.include "common.inc"
  START
  li    t6, 500
  li    a0, 3
  li    a1, 1
  ROI_BEGIN
1:
  .rept 8
  mul   a1, a1, a0
  .endr
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
