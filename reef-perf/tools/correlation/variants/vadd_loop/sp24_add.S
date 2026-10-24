# vec_add_m1 variant 'sp24_add': same loop and addresses as vec_add_m1,
# with only these vector ops in the inner loop. For RTL correlation.
.include "common.inc"
  START
  li    t6, 20
  ROI_BEGIN
1:
  la    a0, vec_a
  la    a1, vec_b
  la    a2, vec_c
  li    a3, 1024
2:
  vsetvli t0, a3, e32, m1, ta, ma
  vadd.vv v3, v1, v2
  addi  s2, s2, 1
  addi  s3, s3, 1
  addi  s4, s4, 1
  addi  s5, s5, 1
  addi  s6, s6, 1
  addi  s7, s7, 1
  addi  s8, s8, 1
  addi  s9, s9, 1
  addi  s2, s2, 1
  addi  s3, s3, 1
  addi  s4, s4, 1
  addi  s5, s5, 1
  addi  s6, s6, 1
  addi  s7, s7, 1
  addi  s8, s8, 1
  addi  s9, s9, 1
  addi  s2, s2, 1
  addi  s3, s3, 1
  addi  s4, s4, 1
  addi  s5, s5, 1
  addi  s6, s6, 1
  addi  s7, s7, 1
  addi  s8, s8, 1
  addi  s9, s9, 1
  slli  t1, t0, 2
  add   a0, a0, t1
  add   a1, a1, t1
  add   a2, a2, t1
  sub   a3, a3, t0
  bnez  a3, 2b
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END

  .data
  .balign 16
vec_a: .rept 1024
  .word 1
  .endr
vec_b: .rept 1024
  .word 2
  .endr
  .bss
  .balign 16
vec_c: .space 4096
