# Dependent ALU chain: each add depends on the previous one. 1000 iterations
# of 16 adds. Measures ALU latency (M3: 1 cycle, back-to-back).
.include "common.inc"
# Override defaults with assembler --defsym NAME=value.
.ifndef ITERATIONS
  .set ITERATIONS, 1000
.endif
.ifndef ADDS
  .set ADDS, 16
.endif
  START
  li    t6, ITERATIONS
  ROI_BEGIN
1:
  .rept ADDS
  add   a0, a0, a1
  .endr
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
