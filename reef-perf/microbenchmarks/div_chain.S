# Dependent divides with a large and a small dividend. On M3 the divider
# skips leading zeros, so the small-dividend divides should be faster.
.include "common.inc"
# Override defaults with assembler --defsym NAME=value.
.ifndef ITERATIONS
  .set ITERATIONS, 200
.endif
.ifndef DIV_GROUPS
  .set DIV_GROUPS, 1
.endif
  START
  li    t6, ITERATIONS
  li    a2, 3
  ROI_BEGIN
1:
  .rept DIV_GROUPS
  li    a0, 0x7fffffff
  divu  a0, a0, a2
  divu  a0, a0, a2
  li    a1, 100
  divu  a1, a1, a2
  divu  a1, a1, a2
  .endr
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
