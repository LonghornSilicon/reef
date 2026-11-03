# Tight loop: 2-instruction body, 5000 iterations. Almost pure branch cost:
# measures the taken-branch (predicted backward-taken) penalty.
.include "common.inc"
# Override defaults with assembler --defsym NAME=value.
.ifndef ITERATIONS
  .set ITERATIONS, 5000
.endif
  START
  li    t6, ITERATIONS
  ROI_BEGIN
1:
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
