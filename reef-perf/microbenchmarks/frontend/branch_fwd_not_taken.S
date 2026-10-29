# Control for branch_bwd_not_taken: same loop, but the never-taken branch is
# forward, so M3 predicts it not taken (correct, no redirect).
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
  bne   x0, x0, 2f      # forward, never taken -> predicted correctly
  bnez  t6, 1b          # loop branch, predicted taken (correct)
  ROI_END
2:
  END
