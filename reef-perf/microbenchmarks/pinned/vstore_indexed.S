# 16 indexed vsuxei8.v per iteration, indices 0 to 15 (one 16-byte line).
.include "common.inc"
.ifndef ITERS
  .equ ITERS, 50
.endif
  START
  li    t0, 16
  vsetvli x0, t0, e8, m1, ta, ma
  vid.v v4
  li    t6, ITERS
  la    a4, buf
  ROI_BEGIN
  .balign 16
1:
  vsuxei8.v v8, (a4), v4
  vsuxei8.v v8, (a4), v4
  vsuxei8.v v8, (a4), v4
  vsuxei8.v v8, (a4), v4
  vsuxei8.v v8, (a4), v4
  vsuxei8.v v8, (a4), v4
  vsuxei8.v v8, (a4), v4
  vsuxei8.v v8, (a4), v4
  vsuxei8.v v8, (a4), v4
  vsuxei8.v v8, (a4), v4
  vsuxei8.v v8, (a4), v4
  vsuxei8.v v8, (a4), v4
  vsuxei8.v v8, (a4), v4
  vsuxei8.v v8, (a4), v4
  vsuxei8.v v8, (a4), v4
  vsuxei8.v v8, (a4), v4
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
  .data
  .balign 16
buf: .space 512
