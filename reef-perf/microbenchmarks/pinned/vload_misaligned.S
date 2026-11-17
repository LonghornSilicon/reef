# 16 vle8.v per iteration from an address 8 bytes into a line: each register
# spans two 16-byte lines.
.include "common.inc"
.ifndef ITERS
  .equ ITERS, 50
.endif
  START
  li    t0, 16
  vsetvli x0, t0, e8, m1, ta, ma
  li    t6, ITERS
  la    a4, buf
  addi  a5, a4, 8
  ROI_BEGIN
  .balign 16
1:
  vle8.v v8, (a5)
  vle8.v v8, (a5)
  vle8.v v8, (a5)
  vle8.v v8, (a5)
  vle8.v v8, (a5)
  vle8.v v8, (a5)
  vle8.v v8, (a5)
  vle8.v v8, (a5)
  vle8.v v8, (a5)
  vle8.v v8, (a5)
  vle8.v v8, (a5)
  vle8.v v8, (a5)
  vle8.v v8, (a5)
  vle8.v v8, (a5)
  vle8.v v8, (a5)
  vle8.v v8, (a5)
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
  .data
  .balign 16
buf: .space 512
