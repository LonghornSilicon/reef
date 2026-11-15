# 16 vluxei8.v, indices 0..15 (contiguous bytes, gathered).
# Generated for the LSU study (docs/lsu-study.md). 16 accesses per
# iteration, unrolled, ITERS times.
.include "common.inc"
.ifndef ITERS
  .equ ITERS, 20
.endif
  START
  li    t0, 16
  vsetvli x0, t0, e8, m1, ta, ma
  vid.v v4
  li    t6, ITERS
  ROI_BEGIN
1:
  la    a4, buf
  vluxei8.v v8, (a4), v4
  addi  a4, a4, 16
  vluxei8.v v9, (a4), v4
  addi  a4, a4, 16
  vluxei8.v v10, (a4), v4
  addi  a4, a4, 16
  vluxei8.v v11, (a4), v4
  addi  a4, a4, 16
  vluxei8.v v12, (a4), v4
  addi  a4, a4, 16
  vluxei8.v v13, (a4), v4
  addi  a4, a4, 16
  vluxei8.v v14, (a4), v4
  addi  a4, a4, 16
  vluxei8.v v15, (a4), v4
  addi  a4, a4, 16
  vluxei8.v v16, (a4), v4
  addi  a4, a4, 16
  vluxei8.v v17, (a4), v4
  addi  a4, a4, 16
  vluxei8.v v18, (a4), v4
  addi  a4, a4, 16
  vluxei8.v v19, (a4), v4
  addi  a4, a4, 16
  vluxei8.v v20, (a4), v4
  addi  a4, a4, 16
  vluxei8.v v21, (a4), v4
  addi  a4, a4, 16
  vluxei8.v v22, (a4), v4
  addi  a4, a4, 16
  vluxei8.v v23, (a4), v4
  addi  a4, a4, 16
  addi  t6, t6, -1
  bnez  t6, 1b
  vmv.x.s t0, v23
  add   t1, t0, t0
  ROI_END
  END

  .data
  .balign 16

buf: .space 256
