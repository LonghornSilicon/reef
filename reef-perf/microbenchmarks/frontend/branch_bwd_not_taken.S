# Backward branch that is never taken. M3 predicts every backward branch
# taken (UncachedFetch.scala:87-100), so fetch follows the target and the BRU
# redirects to pc+4 (Bru.scala:202-212): one mispredict per iteration.
# Control: branch_fwd_not_taken (same shape, predicted correctly).
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
  bne   x0, x0, 1b      # backward, never taken -> mispredict
  bnez  t6, 1b          # loop branch, predicted taken (correct)
  ROI_END
  END
