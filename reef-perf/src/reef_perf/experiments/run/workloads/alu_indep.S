# Independent ALU stream: 16 independent adds per iteration, 1000 iterations.
# Measures the fetch/dispatch ceiling on straight-line code.
# M3 RTL expectation: ~2 IPC (fetch delivers 4 instructions every other cycle).
.include "common.inc"
  START
  li    t6, 1000
  ROI_BEGIN
1:
  .rept 4
  addi  a0, a0, 1
  addi  a1, a1, 1
  addi  a2, a2, 1
  addi  a3, a3, 1
  .endr
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
