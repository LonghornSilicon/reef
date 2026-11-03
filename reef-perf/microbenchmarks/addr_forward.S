# Address produced by an ALU op the cycle before it is used by a load.
# On M3 a load's base register must come from the *registered* scoreboard, so
# `addi a0 -> lw 0(a0)` costs one more cycle than `addi a0 -> add`.
.include "common.inc"
# Override defaults with assembler --defsym NAME=value.
.ifndef ITERATIONS
  .set ITERATIONS, 1000
.endif
.ifndef PAIRS
  .set PAIRS, 8
.endif
  START
  li    t6, ITERATIONS
  la    a0, buf
  ROI_BEGIN
1:
  .rept PAIRS
  addi  a0, a0, 0
  lw    a1, 0(a0)
  .endr
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END

  .data
  .balign 16
buf:
  .word 0, 0, 0, 0
