# Independent stream: 2 x vdiv.vv at LMUL=8 (8 uops each) per iteration, 25
# iterations (roadmap B18). Dividend registers alternate between two 31-bit
# values, so no two consecutive uops reuse the divider's last result.
.include "common.inc"
  START
  vsetivli zero, 4, e32, m1, ta, ma
  li       t0, 0x7fffffff
  li       t1, 0x7ffffffe
  vmv.v.x  v8, t0
  vmv.v.x  v9, t1
  vmv.v.x  v10, t0
  vmv.v.x  v11, t1
  vmv.v.x  v12, t0
  vmv.v.x  v13, t1
  vmv.v.x  v14, t0
  vmv.v.x  v15, t1
  li       t0, 32
  vsetvli  zero, t0, e32, m8, ta, ma
  vmv.v.i  v24, -1
  li       t6, 25
  ROI_BEGIN
  .balign 16            # loop at slot 0 of a 16-byte fetch block
1:
  vdiv.vv  v0, v8, v24
  vdiv.vv  v16, v8, v24
  addi     t6, t6, -1
  bnez     t6, 1b
  ROI_END
  END
