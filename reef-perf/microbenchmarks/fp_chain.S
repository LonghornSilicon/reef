# Dependent FP32 multiply-add chain. Measures FPU latency.
.include "common.inc"
# Override defaults with assembler --defsym NAME=value.
.ifndef ITERATIONS
  .set ITERATIONS, 500
.endif
.ifndef FMADDS
  .set FMADDS, 8
.endif
  START
  li    t6, ITERATIONS
  fmv.w.x f1, zero
  fmv.w.x f2, zero
  fmv.w.x f3, zero
  ROI_BEGIN
1:
  .rept FMADDS
  fmadd.s f3, f1, f2, f3
  .endr
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
