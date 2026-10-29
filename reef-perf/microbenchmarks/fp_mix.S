# FP32 adds mixed with integer adds: [fadd.s, addi, addi, addi].
# On M3 every F instruction dispatches alone from slot 0, so the integer adds
# cannot share its cycle.
.include "common.inc"
# Override defaults with assembler --defsym NAME=value.
.ifndef ITERATIONS
  .set ITERATIONS, 1000
.endif
.ifndef GROUPS
  .set GROUPS, 4
.endif
  START
  li    t6, ITERATIONS
  fmv.w.x f1, zero
  fmv.w.x f2, zero
  ROI_BEGIN
1:
  .rept GROUPS
  fadd.s f3, f1, f2
  addi  a0, a0, 1
  addi  a1, a1, 1
  addi  a2, a2, 1
  .endr
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
