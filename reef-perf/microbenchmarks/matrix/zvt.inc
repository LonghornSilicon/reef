# Zvt (CoralNPU matrix extension, spec v0.4) instructions as .insn, since the
# assembler does not know them. Encodings: RvvDecode.scala:178-376 and
# tests/cocotb/vme_test/vme_test_utils.h at CoralNPU 8a5e1b31. Register
# operands written as xN only fill an encoding field with N.

# MS_ON: mstatus.MS = Initial (bits 30:29, Csr.scala:626-627). With MS = Off
# every tile instruction is illegal (Decode.scala:591-596). CoralNPU resets MS
# to Initial (Csr.scala:355), so this only guards against software that
# turned it off. Uses t0.
.macro MS_ON
  li    t0, (1 << 29)
  csrs  mstatus, t0
.endm

# msetmtype rs1, rs2: mtype <- rs1, vtype <- rs2, vl <- 0.
.macro MSETMTYPE rs1, rs2
  .insn r 0x57, 7, 0x42, x0, \rs1, \rs2
.endm
# msettn rd, rs1: vl <- min(rs1, VLMAX).
.macro MSETTN rd, rs1
  .insn r 0x57, 7, 0x44, \rd, \rs1, x0
.endm
# msettm rd, rs1 / msettk rd, rs1.
.macro MSETTM rd, rs1
  .insn r 0x57, 7, 0x44, \rd, \rs1, x1
.endm
.macro MSETTK rd, rs1
  .insn r 0x57, 7, 0x44, \rd, \rs1, x2
.endm

# vtzero mt<tile>: rd field = tile << 1. Pass the field as xN (mt4 -> x8).
.macro VTZERO rdfield
  .insn r 0x57, 6, 0x21, \rdfield, x0, x30
.endm
# vtmmu.tvv mt<tile>, vs2, vs1 (int8 x int8 -> int32, A unsigned).
.macro VTMMU rdfield, vs2, vs1
  .insn r 0x77, 0, 0x79, \rdfield, \vs1, \vs2
.endm
# vtmv.v.t vd, rs1 (rs1 = TSS).
.macro VTMV_VT vd, rs1
  .insn r 0x57, 6, 0x21, \vd, \rs1, x31
.endm
# vtmv.t.v rs1, vs2 (rs1 = TSS).
.macro VTMV_TV rs1, vs2
  .insn r 0x57, 6, 0x2f, x0, \rs1, \vs2
.endm
# vtle8 / vtse8 / vtle32 / vtse32 (rs1 = address, rs2 = TSS).
.macro VTLE8 rs1, rs2
  .insn r 0x07, 7, 0x09, x0, \rs1, \rs2
.endm
.macro VTSE8 rs1, rs2
  .insn r 0x27, 7, 0x09, x0, \rs1, \rs2
.endm
.macro VTLE32 rs1, rs2
  .insn r 0x07, 7, 0x29, x0, \rs1, \rs2
.endm
.macro VTSE32 rs1, rs2
  .insn r 0x27, 7, 0x29, x0, \rs1, \rs2
.endm

# mtype values: tm[23:10] | tk[7:5] | mtwiden[1:0].
.equ MTYPE_I8_I32, (16 << 10) | (4 << 5) | 3  # tm 16, tk 4, x4: int8 MAC
.equ MTYPE_E32,    (16 << 10) | (1 << 5) | 1  # tm 16, tk 1, x1: 32-bit tiles
.equ MTYPE_E8,     (16 << 10) | (0 << 5) | 1  # tm 16, x1: 8-bit tiles
.equ VTYPE_E8,  0                             # SEW8 (LMUL derived)
.equ VTYPE_E32, (2 << 3)                      # SEW32 (LMUL derived)
