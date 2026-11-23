# Pointer chase through vle8.v: vle8 -> vmv.x.s -> add into the next
# address. Measures load-to-use latency, not throughput.
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
  vle8.v v8, (a4)
  vmv.x.s t0, v8
  add   a4, a4, t0
  vle8.v v8, (a4)
  vmv.x.s t0, v8
  add   a4, a4, t0
  vle8.v v8, (a4)
  vmv.x.s t0, v8
  add   a4, a4, t0
  vle8.v v8, (a4)
  vmv.x.s t0, v8
  add   a4, a4, t0
  vle8.v v8, (a4)
  vmv.x.s t0, v8
  add   a4, a4, t0
  vle8.v v8, (a4)
  vmv.x.s t0, v8
  add   a4, a4, t0
  vle8.v v8, (a4)
  vmv.x.s t0, v8
  add   a4, a4, t0
  vle8.v v8, (a4)
  vmv.x.s t0, v8
  add   a4, a4, t0
  vle8.v v8, (a4)
  vmv.x.s t0, v8
  add   a4, a4, t0
  vle8.v v8, (a4)
  vmv.x.s t0, v8
  add   a4, a4, t0
  vle8.v v8, (a4)
  vmv.x.s t0, v8
  add   a4, a4, t0
  vle8.v v8, (a4)
  vmv.x.s t0, v8
  add   a4, a4, t0
  vle8.v v8, (a4)
  vmv.x.s t0, v8
  add   a4, a4, t0
  vle8.v v8, (a4)
  vmv.x.s t0, v8
  add   a4, a4, t0
  vle8.v v8, (a4)
  vmv.x.s t0, v8
  add   a4, a4, t0
  vle8.v v8, (a4)
  vmv.x.s t0, v8
  add   a4, a4, t0
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END

  .data
  .balign 16

buf: .space 16
