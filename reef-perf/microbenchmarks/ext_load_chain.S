# Pointer chase in external memory: each load's address comes from the
# previous load, so no two loads overlap. Measures the latency of one load
# over the e-bus and AXI master. 64-node circular list at 0x20000000 (the M3
# testbench's memory, tests/systemc/Xbar.h), 500 steps.
.include "common.inc"
  START
  # Build the list with stores before the ROI: node[i] = &node[(i+1) % 64],
  # one node per 16-byte line.
  li    a0, 0x20000000
  li    t0, 63
  mv    a1, a0
1:
  addi  a2, a1, 16
  sw    a2, 0(a1)
  mv    a1, a2
  addi  t0, t0, -1
  bnez  t0, 1b
  sw    a0, 0(a1)           # last node points back to the first

  li    t6, 500
  ROI_BEGIN
2:
  lw    a0, 0(a0)
  addi  t6, t6, -1
  bnez  t6, 2b
  ROI_END
  END
