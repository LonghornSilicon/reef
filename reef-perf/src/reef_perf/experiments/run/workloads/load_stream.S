# Independent word loads walking a 1 KB array, 8 loads per iteration.
# Measures LSU throughput. On M3 the LSU handles one memory instruction at a
# time; consecutive loads to the same 16-byte line still take separate turns.
.include "common.inc"
  START
  li    t6, 256
  ROI_BEGIN
1:
  la    a0, array
  li    t4, 32              # 32 x 8 words = 1 KB
2:
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
  bnez  t4, 2b
  addi  t6, t6, -32
  bgtz  t6, 1b
  ROI_END
  END

  .data
  .balign 16
array:
  .rept 256
  .word 1
  .endr
