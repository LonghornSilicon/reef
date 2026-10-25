# Same as vec_add_m1 but LMUL=4 (16 elements/op): fewer, longer vector ops.
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
  vsetvli t0, a3, e32, m4, ta, ma
  vle32.v v4, (a0)
  vle32.v v8, (a1)
  vadd.vv v12, v4, v8
  vse32.v v12, (a2)
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
