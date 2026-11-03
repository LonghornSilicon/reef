# Not-taken branches inside the loop body: [alu, beq(not taken), alu, alu].
# On M3 a conditional branch ends the dispatch group, so this should dispatch
# ~2 instructions per cycle at best, versus ~4 if that rule did not exist.
.include "common.inc"
# Override defaults with assembler --defsym NAME=value.
.ifndef ITERATIONS
  .set ITERATIONS, 1000
.endif
.ifndef GROUPS
  .set GROUPS, 4
.endif
  START
  li    t6, ITERATIONS
  li    t5, 1
  ROI_BEGIN
1:
  .rept GROUPS
  addi  a0, a0, 1
  beqz  t5, 2f          # never taken (t5 = 1)
  addi  a1, a1, 1
  addi  a2, a2, 1
  .endr
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
2:
  END
