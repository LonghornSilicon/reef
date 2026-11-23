# 8 vle8.v + vse8.v pairs at LMUL 1 (16 accesses), like vec_memcpy.
# Generated for the LSU study (docs/lsu-study.md). 16 accesses per
# iteration, unrolled, ITERS times.
.include "common.inc"
.ifndef ITERS
  .equ ITERS, 20
.endif
  START
  li    t0, 16
  vsetvli x0, t0, e8, m1, ta, ma
  li    t6, ITERS
  ROI_BEGIN
1:
  la    a4, buf
  la    a5, buf + 128
  vle8.v v8, (a4)
  vse8.v v8, (a5)
  addi  a4, a4, 16
  addi  a5, a5, 16
  vle8.v v9, (a4)
  vse8.v v9, (a5)
  addi  a4, a4, 16
  addi  a5, a5, 16
  vle8.v v10, (a4)
  vse8.v v10, (a5)
  addi  a4, a4, 16
  addi  a5, a5, 16
  vle8.v v11, (a4)
  vse8.v v11, (a5)
  addi  a4, a4, 16
  addi  a5, a5, 16
  vle8.v v12, (a4)
  vse8.v v12, (a5)
  addi  a4, a4, 16
  addi  a5, a5, 16
  vle8.v v13, (a4)
  vse8.v v13, (a5)
  addi  a4, a4, 16
  addi  a5, a5, 16
  vle8.v v14, (a4)
  vse8.v v14, (a5)
  addi  a4, a4, 16
  addi  a5, a5, 16
  vle8.v v15, (a4)
  vse8.v v15, (a5)
  addi  a4, a4, 16
  addi  a5, a5, 16
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END

  .data
  .balign 16

buf: .space 256
