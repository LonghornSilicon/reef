# The inner loop of a quantized linear layer (reef_kernels::linear_bias_int8),
# ITERS times: a vle8.v, a scalar lb, a vsext.vf2 of the loaded register and
# a vwmacc.vx into an e32 m4 accumulator, at e16 m2. The accumulator chain
# backs the vector pipeline up, so the load's uop is dispatched late and the
# LSU has to wait for the vector backend to take its data.
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
  vle8.v v1, (t4)
  lb    t5, 0(t1)
  addi  t1, t1, 1
  add   t4, t4, a2
  vsext.vf2 v2, v1
  vwmacc.vx v4, t5, v2
  bne   t6, t1, 1b
  vmv.x.s a0, v4
  add   a1, a0, a0
  ROI_END
  END
  .data
  .balign 16
buf: .space 256
