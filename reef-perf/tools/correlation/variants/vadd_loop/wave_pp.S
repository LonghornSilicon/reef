# Scalar <-> vector ping-pong (B17): every iteration crosses into the vector
# unit and back, and each crossing depends on the one before.
#
# Per iteration:  vsetivli -> vmv.s.x (scalar -> vector) -> vredsum.vs
#                 -> vmv.x.s (vector -> scalar) -> addi on the result,
#                 which feeds the next iteration's vmv.s.x.
# vl = 1, so the reduction only reads element 0 and the values are the same
# on Spike and the RTL.
.include "common.inc"
  START
  li    t6, 8
  li    t0, 1
  ROI_BEGIN
1:
  vsetivli zero, 1, e32, m1, ta, ma
  vmv.s.x  v1, t0
  vredsum.vs v2, v1, v1
  vmv.x.s  t0, v2
  addi  t0, t0, 1
  addi  t6, t6, -1
  bnez  t6, 1b
  ROI_END
  END
