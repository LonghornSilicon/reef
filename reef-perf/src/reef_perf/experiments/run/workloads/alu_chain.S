# Dependent ALU chain: each add depends on the previous one. 1000 iterations
# of 16 adds. Measures ALU latency (M3: 1 cycle, back-to-back).
.include "common.inc"
  START
  li    t6, 1000
  ROI_BEGIN
1:
  .rept 16
  add   a0, a0, a1
  .endr
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
