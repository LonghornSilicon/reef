# DMA queue demo. Not an M3 microbenchmark: M3 has no DMA, so there are no RTL
# numbers. It starts two 4 KB transfers from external memory into the DTCM
# back to back, then runs a loop that does not touch memory, then waits for
# both. With dma_queue_depth 1 the second GO waits for the first transfer,
# so the loop starts later (tests/python/integration/test_dma_queue.py).
.include "common.inc"
  START
  li    s0, 0x40000000        # DMA register block
  la    s1, tile
  ROI_BEGIN
  li    t0, 0x20000000        # SRC: external memory
  sw    t0, 0x0(s0)
  sw    s1, 0x4(s0)           # DST: tile, in the DTCM
  li    t0, 4096
  sw    t0, 0x8(s0)           # LEN
  sw    zero, 0xc(s0)         # GO
  sw    zero, 0xc(s0)         # GO again, same transfer
  li    t5, 150
1:
  addi  t5, t5, -1
  bnez  t5, 1b
  lw    t3, 0x10(s0)          # WAIT: its data comes when both are done
  add   t4, t3, t3            # Use it, so the core stalls until then
  ROI_END
  END

  .bss
  .balign 16
tile: .space 4096
