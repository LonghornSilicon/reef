# vec_add_m1 variant 'nv_addonly': vsetvli hoisted out of the loop (vl = 4 for the
# whole ROI, t0 = 4), so the inner loop has only these vector ops. For RTL correlation.
.include "common.inc"
  START
  vsetivli t0, 4, e32, m1, ta, ma
  li    t6, 20
  ROI_BEGIN
1:
  la    a0, vec_a
  la    a1, vec_b
  la    a2, vec_c
  li    a3, 1024
2:
  vadd.vv v3, v1, v2
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
