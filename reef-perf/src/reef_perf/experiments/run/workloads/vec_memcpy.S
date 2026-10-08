# Copy 4 KB with vle8/vse8 at LMUL=8 (128 bytes per op). 10 passes.
# Pure vector load/store bandwidth.
.include "common.inc"
  START
  li    t6, 10
  ROI_BEGIN
1:
  la    a0, src
  la    a1, dst
  li    a2, 4096
2:
  vsetvli t0, a2, e8, m8, ta, ma
  vle8.v v8, (a0)
  vse8.v v8, (a1)
  add   a0, a0, t0
  add   a1, a1, t0
  sub   a2, a2, t0
  bnez  a2, 2b
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END

  .data
  .balign 16
src: .rept 1024
  .word 0x01020304
  .endr
  .bss
  .balign 16
dst: .space 4096
