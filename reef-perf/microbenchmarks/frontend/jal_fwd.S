# Control for branch_fwd_taken and jalr_chain: same loop, but the jump is a
# jal, which M3 predicts taken (UncachedFetch.scala:87-100): no redirect.
.include "common.inc"
# Override defaults with assembler --defsym NAME=value.
.ifndef ITERATIONS
  .set ITERATIONS, 1000
.endif
  START
  li    t6, ITERATIONS
  ROI_BEGIN
  .balign 16            # loop at slot 0 of a 16-byte fetch block
1:
  addi  t6, t6, -1
  jal   x0, 2f          # forward jal, predicted taken (correct)
  nop                   # skipped
2:
  bnez  t6, 1b          # loop branch, predicted taken (correct)
  ROI_END
  END
