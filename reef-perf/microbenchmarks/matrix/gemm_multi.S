# A 32x32 int8 GEMM with K = 64 as four 16x16 output tiles: each of 16
# k-steps loads A (32 x 4 k-rows, two 16-row halves) and B (4 k-rows x 32,
# two halves) with vle8.v and runs four vtmmu, one per accumulator group
# (mt0, mt4, mt8, mt12); then the four int32 tiles are read back with
# vtmv.v.t + vse32. The whole GEMM runs ITERS (1) times. Held out: written
# after the model was calibrated; nothing is fitted to it.
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
  VTZERO x0
  VTZERO x8
  VTZERO x16
  VTZERO x24

  li    a0, MTYPE_I8_I32
  li    a1, VTYPE_E8
  MSETMTYPE a0, a1
  li    a0, 16
  MSETTN x0, a0
  la    a2, mat_a
  la    a3, mat_b
  li    t6, 16
1:
  vle8.v v8, (a2)
  addi  a2, a2, 16
  vle8.v v10, (a2)
  addi  a2, a2, 16
  vle8.v v12, (a2)
  addi  a2, a2, 16
  vle8.v v14, (a2)
  addi  a2, a2, 16
  vle8.v v24, (a2)
  addi  a2, a2, 16
  vle8.v v26, (a2)
  addi  a2, a2, 16
  vle8.v v28, (a2)
  addi  a2, a2, 16
  vle8.v v30, (a2)
  addi  a2, a2, 16
  vle8.v v16, (a3)
  addi  a3, a3, 16
  vle8.v v18, (a3)
  addi  a3, a3, 16
  vle8.v v20, (a3)
  addi  a3, a3, 16
  vle8.v v22, (a3)
  addi  a3, a3, 16
  vle8.v v0, (a3)
  addi  a3, a3, 16
  vle8.v v2, (a3)
  addi  a3, a3, 16
  vle8.v v4, (a3)
  addi  a3, a3, 16
  vle8.v v6, (a3)
  addi  a3, a3, 16
  VTMMU x0, x8, x16     # C00 (mt0-mt3)   += A0^T B0
  VTMMU x8, x8, x0      # C01 (mt4-mt7)   += A0^T B1
  VTMMU x16, x24, x16   # C10 (mt8-mt11)  += A1^T B0
  VTMMU x24, x24, x0    # C11 (mt12-mt15) += A1^T B1
  addi  t6, t6, -1
  bnez  t6, 1b

  li    a0, MTYPE_E32
  li    a1, VTYPE_E32
  MSETMTYPE a0, a1
  li    a0, 16
  MSETTN x0, a0
  la    a1, mat_c
  li    a0, (0 << 27)  # TSS: tile 0, row 0
  li    t6, 16
20:
  VTMV_VT x4, a0        # v4..v7 <- row of mt0-mt3
  vse32.v v4, (a1)
  addi  a1, a1, 64
  addi  a0, a0, 1
  addi  t6, t6, -1
  bnez  t6, 20b
  li    a0, (4 << 27)  # TSS: tile 4, row 0
  li    t6, 16
24:
  VTMV_VT x4, a0        # v4..v7 <- row of mt4-mt7
  vse32.v v4, (a1)
  addi  a1, a1, 64
  addi  a0, a0, 1
  addi  t6, t6, -1
  bnez  t6, 24b
  li    a0, (8 << 27)  # TSS: tile 8, row 0
  li    t6, 16
28:
  VTMV_VT x4, a0        # v4..v7 <- row of mt8-mt11
  vse32.v v4, (a1)
  addi  a1, a1, 64
  addi  a0, a0, 1
  addi  t6, t6, -1
  bnez  t6, 28b
  li    a0, (12 << 27)  # TSS: tile 12, row 0
  li    t6, 16
32:
  VTMV_VT x4, a0        # v4..v7 <- row of mt12-mt15
  vse32.v v4, (a1)
  addi  a1, a1, 64
  addi  a0, a0, 1
  addi  t6, t6, -1
  bnez  t6, 32b
  addi  s2, s2, -1
  bnez  s2, 0b
  # Read the last row back so ROI_END waits for it.
  vmv.x.s t0, v4
  add   t1, t0, t0
  ROI_END
  END

  .data
  .balign 16
mat_a: .rept 2048
  .byte 1
  .endr
mat_b: .rept 2048
  .byte 2
  .endr
  .bss
  .balign 16
mat_c: .space 4096
