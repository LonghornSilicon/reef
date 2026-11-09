# One 16x16 int8 tile GEMM with K = 256 (64 k-steps), then the int32 result
# read back; otherwise gemm_tile. Held out: nothing is fitted to it.
# (From gemm_tile:) One 16x16 int8 tile GEMM with K = 64, then the int32 result read back,
# following vme_matmul_test.cc: clear the accumulator, then 16 times load A
# and B (4 k-rows each) with vle8.v and vtmmu; finally 16 x (vtmv.v.t, vse32).
# Exercises vector-register dependencies into and out of the matrix module.
# The whole GEMM runs ITERS (1) times; matrix_diff builds N and 2N.
.include "common.inc"
.include "matrix/zvt.inc"
.ifndef ITERS
  .equ ITERS, 1          # GEMM count; override with --defsym ITERS=<n>
.endif
  START
  MS_ON
  li    s2, ITERS
  ROI_BEGIN
0:
  li    a0, MTYPE_E32
  li    a1, VTYPE_E32
  MSETMTYPE a0, a1
  li    a0, 16
  MSETTN x0, a0
  VTZERO x0             # mt0-mt3

  li    a0, MTYPE_I8_I32
  li    a1, VTYPE_E8
  MSETMTYPE a0, a1
  li    a0, 16
  MSETTN x0, a0
  la    a2, mat_a
  la    a3, mat_b
  li    t6, 64
1:
  vle8.v v8, (a2)
  addi  a2, a2, 16
  vle8.v v10, (a2)
  addi  a2, a2, 16
  vle8.v v12, (a2)
  addi  a2, a2, 16
  vle8.v v14, (a2)
  addi  a2, a2, 16
  vle8.v v16, (a3)
  addi  a3, a3, 16
  vle8.v v18, (a3)
  addi  a3, a3, 16
  vle8.v v20, (a3)
  addi  a3, a3, 16
  vle8.v v22, (a3)
  addi  a3, a3, 16
  VTMMU x0, x8, x16     # mt0 += A[k..k+3]^T B[k..k+3]
  addi  t6, t6, -1
  bnez  t6, 1b

  li    a0, MTYPE_E32
  li    a1, VTYPE_E32
  MSETMTYPE a0, a1
  li    a0, 16
  MSETTN x0, a0
  la    a1, mat_c
  li    a0, 0           # TSS: tile 0, row 0
  li    t6, 16
2:
  VTMV_VT x4, a0        # v4..v7 <- row of mt0-mt3
  vse32.v v4, (a1)
  addi  a1, a1, 64
  addi  a0, a0, 1
  addi  t6, t6, -1
  bnez  t6, 2b
  addi  s2, s2, -1
  bnez  s2, 0b
  # Read the last row back so ROI_END waits for it (the vse32 cannot be
  # waited for, open case 2).
  vmv.x.s t0, v4
  add   t1, t0, t0
  ROI_END
  END

  .data
  .balign 16
mat_a: .rept 4096
  .byte 1
  .endr
mat_b: .rept 4096
  .byte 2
  .endr
  .bss
  .balign 16
mat_c: .space 1024
