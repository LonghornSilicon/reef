# Tile instructions the RVV decoder rejects retire as no-ops (rvv_backend_
# decode_unit.sv:92-96): here a vtle8 and a vtmmu with vl = tn = 0
# (rvv_backend_decode_unit_lsu.sv:3387-3393, _ari.sv:3717-3722).
.include "common.inc"
.include "matrix/zvt.inc"
  START
  MS_ON
  li    a0, MTYPE_I8_I32
  li    a1, VTYPE_E8
  MSETMTYPE a0, a1        # vl = 0
  la    a4, buf
  li    a5, 0
  ROI_BEGIN
  VTLE8 a4, a5            # vl = 0: no-op
  VTMMU x0, x8, x16       # tn = 0: no-op
  ROI_END
  END

  .data
  .balign 16
buf: .space 16
