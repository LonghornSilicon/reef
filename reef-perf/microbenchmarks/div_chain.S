# Dependent divides with a large and a small dividend. On M3 the divider
# skips leading zeros, so the small-dividend divides should be faster.
.include "common.inc"
  START
  li    t6, 200
  li    a2, 3
  ROI_BEGIN
1:
  li    a0, 0x7fffffff
  divu  a0, a0, a2
  divu  a0, a0, a2
  li    a1, 100
  divu  a1, a1, a2
  divu  a1, a1, a2
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
