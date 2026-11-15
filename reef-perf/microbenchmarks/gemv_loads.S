# The loads of gemv_int8 alone: the same loops and addresses, but each row
# only does its vle8 (64 B, LMUL = 4, so 4 vector registers). Isolates the
# cost of vector loads (B14) from the vector arithmetic (B15).
.include "common.inc"
  START
  li    t6, 4
  ROI_BEGIN
1:
  la    a0, W
  la    a1, x
  la    a2, y
  li    a3, 64              # rows
  li    a4, 64              # K
  vsetvli zero, a4, e8, m4, ta, ma
  vle8.v  v4, (a1)          # x
2:
  vsetvli zero, a4, e8, m4, ta, ma
  vle8.v  v8, (a0)          # one row of W
  addi  a0, a0, 64
  addi  a2, a2, 4
  addi  a3, a3, -1
  bnez  a3, 2b
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END

  .data
  .balign 16
W: .rept 64 * 16
  .byte 1, 2, 3, 4
  .endr
x: .rept 16
  .byte 1, 1, 1, 1
  .endr
  .bss
  .balign 16
y: .space 64 * 4
