# Tight loop: 2-instruction body, 5000 iterations. Almost pure branch cost:
# measures the taken-branch (predicted backward-taken) penalty.
.include "common.inc"
  START
  li    t6, 5000
  ROI_BEGIN
1:
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
