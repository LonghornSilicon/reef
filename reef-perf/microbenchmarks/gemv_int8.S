# int8 GEMV: y[n] = sum_k W[n][k] * x[k], N = 64 rows, K = 64.
# The shape of an LLM decode step (one token, int8 weights) at toy size.
#
# Per row:  vle8 (64 B of W)  ->  vwmul.vv (int8 x int8 -> int16, 64 lanes)
#           -> vwredsum.vs (int16 -> int32 sum)  ->  vmv.x.s  ->  sw
# x (64 B) stays in v4..v7 for the whole kernel. 4 passes over W.
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
  vwmul.vv v16, v8, v4      # int16 products, v16..v23
  vsetvli zero, a4, e16, m8, ta, ma
  vmv.s.x  v0, zero         # reduction seed = 0
  vwredsum.vs v24, v16, v0  # int32 sum in v24[0]
  vsetivli zero, 1, e32, m1, ta, ma
  vmv.x.s  t0, v24
  sw    t0, 0(a2)
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
