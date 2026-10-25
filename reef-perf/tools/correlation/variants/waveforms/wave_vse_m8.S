# Store aliasing check: vse32.v at LMUL=8 (8 registers) in a 3-instruction
# loop, 4 iterations, so several instances of the same store PC are in the
# scalar retirement buffer at once.
.include "common.inc"
  START
  la       a0, buf
  li       t0, 32
  vsetvli  zero, t0, e32, m8, ta, ma
  vmv.v.i  v8, 1
  li       t6, 4
  ROI_BEGIN
  .balign 16
1:
  vse32.v  v8, (a0)
  addi     t6, t6, -1
  bnez     t6, 1b
  ROI_END
  END
  .bss
  .balign 16
buf: .space 128
