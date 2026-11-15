# Independent word loads from external memory, 8 per iteration, 256 in all.
# If the e-bus serves one access at a time, each load waits for the last
# one's response and this runs at ext_load_chain's speed; if accesses
# overlap, it runs faster. Memory at 0x20000000 (the M3 testbench's memory).
.include "common.inc"
  START
  li    a0, 0x20000000
  li    t4, 32              # 32 x 8 loads
  ROI_BEGIN
1:
  lw    a1, 0(a0)
  lw    a2, 4(a0)
  lw    a3, 8(a0)
  lw    a4, 12(a0)
  lw    a5, 16(a0)
  lw    a6, 20(a0)
  lw    a7, 24(a0)
  lw    s2, 28(a0)
  addi  a0, a0, 32
  addi  t4, t4, -1
  bnez  t4, 1b
  ROI_END
  END
