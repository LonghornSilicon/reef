# A vfadd.vf (reads an f register) behind an fadd.s, 4 pairs per iteration:
# the scalar FP result feeding a vector operation.
.include "common.inc"
.ifndef ITERS
  .equ ITERS, 50
.endif
  START
  li    t0, 0x40400000
  fmv.w.x f1, t0
  li    t0, 0x3fc00000
  fmv.w.x f2, t0
  fmv.w.x f3, t0
  li    t0, 4
  vsetvli x0, t0, e32, m1, ta, ma
  vfmv.v.f v8, f1
  li    t6, ITERS
  la    a4, buf
  ROI_BEGIN
  .balign 16
1:
  fadd.s f3, f1, f2
  vfadd.vf v9, v8, f3
  fadd.s f3, f1, f2
  vfadd.vf v9, v8, f3
  fadd.s f3, f1, f2
  vfadd.vf v9, v8, f3
  fadd.s f3, f1, f2
  vfadd.vf v9, v8, f3
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
  .data
  .balign 16
buf: .space 512
