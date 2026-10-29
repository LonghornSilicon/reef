# Dependent multiply chain (500 x 8 muls). Measures MLU latency.
.include "common.inc"
# Override defaults with assembler --defsym NAME=value.
.ifndef ITERATIONS
  .set ITERATIONS, 500
.endif
.ifndef MULS
  .set MULS, 8
.endif
  START
  li    t6, ITERATIONS
  li    a0, 3
  li    a1, 1
  ROI_BEGIN
1:
  .rept MULS
  mul   a1, a1, a0
  .endr
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
