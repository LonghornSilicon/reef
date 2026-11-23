# One vwmacc.vx into an e32 m4 accumulator per iteration, at e16 m2: the
# accumulator chain of a widening multiply-accumulate with 4 uops.
.include "common.inc"
.ifndef ITERS
  .equ ITERS, 50
.endif
  START
  li    t0, 16
  vsetvli x0, t0, e32, m4, ta, ma
  vmv.v.i v4, 0
  vsetvli x0, t0, e16, m2, ta, ma
  la    t4, buf
  la    t1, buf
  addi  t6, t1, ITERS
  li    a2, 0
  ROI_BEGIN
  .balign 16
1:
  addi  t1, t1, 1
  add   t4, t4, a2
  vwmacc.vx v4, t5, v2
  bne   t6, t1, 1b
  vmv.x.s a0, v4
  add   a1, a0, a0
  ROI_END
  END
  .data
  .balign 16
buf: .space 256
