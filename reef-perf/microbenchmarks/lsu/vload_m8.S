# 16 unit-stride vle8.v at LMUL 8 (8 x 16 bytes each).
# Generated for the LSU study (docs/lsu-study.md). 16 accesses per
# iteration, unrolled, ITERS times.
.include "common.inc"
.ifndef ITERS
  .equ ITERS, 20
.endif
  START
  li    t0, 128
  vsetvli x0, t0, e8, m8, ta, ma
  li    t6, ITERS
  ROI_BEGIN
1:
  la    a4, buf
  vle8.v v8, (a4)
  addi  a4, a4, 128
  vle8.v v16, (a4)
  addi  a4, a4, 128
  vle8.v v24, (a4)
  addi  a4, a4, 128
  vle8.v v8, (a4)
  addi  a4, a4, 128
  vle8.v v16, (a4)
  addi  a4, a4, 128
  vle8.v v24, (a4)
  addi  a4, a4, 128
  vle8.v v8, (a4)
  addi  a4, a4, 128
  vle8.v v16, (a4)
  addi  a4, a4, 128
  vle8.v v24, (a4)
  addi  a4, a4, 128
  vle8.v v8, (a4)
  addi  a4, a4, 128
  vle8.v v16, (a4)
  addi  a4, a4, 128
  vle8.v v24, (a4)
  addi  a4, a4, 128
  vle8.v v8, (a4)
  addi  a4, a4, 128
  vle8.v v16, (a4)
  addi  a4, a4, 128
  vle8.v v24, (a4)
  addi  a4, a4, 128
  vle8.v v8, (a4)
  addi  a4, a4, 128
  addi  t6, t6, -1
  bnez  t6, 1b
  vmv.x.s t0, v8
  add   t1, t0, t0
  ROI_END
  END

  .data
  .balign 16

buf: .space 2048
