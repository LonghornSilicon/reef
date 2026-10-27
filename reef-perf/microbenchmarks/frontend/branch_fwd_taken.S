# Forward branch that is always taken. M3 predicts forward branches not taken
# (UncachedFetch.scala:87-100), so the BRU redirects to the target
# (Bru.scala:202-212): one mispredict per iteration.
# Control: jal_fwd (same shape, predicted correctly).
.include "common.inc"
  START
  li    t6, 1000
  ROI_BEGIN
  .balign 16            # loop at slot 0 of a 16-byte fetch block
1:
  addi  t6, t6, -1
  beq   x0, x0, 2f      # forward, always taken -> mispredict
  nop                   # skipped
2:
  bnez  t6, 1b          # loop branch, predicted taken (correct)
  ROI_END
  END
