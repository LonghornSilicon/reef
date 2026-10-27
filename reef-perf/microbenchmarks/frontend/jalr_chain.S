# jalr every iteration. M3 never predicts jalr (no BTB/RAS), so the BRU
# redirects to the register target (Bru.scala:202-212): one mispredict per
# iteration. t0 is set before the loop, so the jalr has no data hazard.
# Control: jal_fwd (same shape, predicted correctly).
.include "common.inc"
  START
  li    t6, 1000
  la    t0, 2f
  ROI_BEGIN
  .balign 16            # loop at slot 0 of a 16-byte fetch block
1:
  addi  t6, t6, -1
  jalr  x0, 0(t0)       # indirect jump, never predicted -> mispredict
  nop                   # skipped
2:
  bnez  t6, 1b          # loop branch, predicted taken (correct)
  ROI_END
  END
