# Dependent FP32 multiply-add chain. Measures FPU latency.
.include "common.inc"
  START
  li    t6, 500
  fmv.w.x f1, zero
  fmv.w.x f2, zero
  fmv.w.x f3, zero
  ROI_BEGIN
1:
  .rept 8
  fmadd.s f3, f1, f2, f3
  .endr
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
