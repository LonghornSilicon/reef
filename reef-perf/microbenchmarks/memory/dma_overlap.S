# DMA demo. Not an M3 microbenchmark: M3 has no DMA, so there are no RTL
# numbers. It starts a 16 KB transfer from HyperRAM into the DTCM, runs 256
# loads from another DTCM buffer while the transfer moves, then waits for it.
# Set LEN to 0 to time the loop alone.
.include "common.inc"
  START
  li    s0, 0x20000000        # DMA register block
  la    s1, tile
  la    s2, work
  ROI_BEGIN
  li    t0, 0x80000000        # SRC: HyperRAM
  sw    t0, 0x0(s0)
  sw    s1, 0x4(s0)           # DST: tile, in the DTCM
  li    t0, 16384
  sw    t0, 0x8(s0)           # LEN
  sw    zero, 0xc(s0)         # GO
  li    t1, 256
1:
  lw    t2, 0(s2)
  addi  s2, s2, 16
  addi  t1, t1, -1
  bnez  t1, 1b
  lw    t3, 0x10(s0)          # WAIT: its data comes when the transfer is done
  add   t4, t3, t3            # Use it, so the core stalls until then
  ROI_END
  END

  .bss
  .align 4
tile: .space 16384
work: .space 4096
