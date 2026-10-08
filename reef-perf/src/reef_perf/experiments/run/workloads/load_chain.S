# Pointer chase: each load's address comes from the previous load.
# Measures load-to-use latency. 64-node circular list, 2000 steps.
.include "common.inc"
  START
  # Build the list: node[i] = &node[(i+1) % 64], one node per 16-byte line.
  la    a0, nodes
  li    t0, 63
  mv    a1, a0
1:
  addi  a2, a1, 16
  sw    a2, 0(a1)
  mv    a1, a2
  addi  t0, t0, -1
  bnez  t0, 1b
  sw    a0, 0(a1)           # last node points back to the first

  li    t6, 2000
  ROI_BEGIN
2:
  lw    a0, 0(a0)
  addi  t6, t6, -1
  bnez  t6, 2b
  ROI_END
  END

  .bss
  .balign 16
nodes:
  .space 64 * 16
