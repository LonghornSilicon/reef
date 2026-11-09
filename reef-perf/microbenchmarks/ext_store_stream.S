# Independent word stores to external memory, 8 per iteration, 256 in all.
# If a store waits for the write response, this costs about as much per
# store as ext_load_chain per load; if stores are posted (the core moves on
# once the write is sent), it costs much less. Memory at 0x20000000 (the M3
# testbench's memory).
.include "common.inc"
  START
  li    a0, 0x20000000
  li    t4, 32              # 32 x 8 stores
  ROI_BEGIN
1:
  sw    t4, 0(a0)
  sw    t4, 4(a0)
  sw    t4, 8(a0)
  sw    t4, 12(a0)
  sw    t4, 16(a0)
  sw    t4, 20(a0)
  sw    t4, 24(a0)
  sw    t4, 28(a0)
  addi  a0, a0, 32
  addi  t4, t4, -1
  bnez  t4, 1b
  ROI_END
  END
