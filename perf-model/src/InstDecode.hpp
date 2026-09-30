// Classifies an instruction and works out which registers it reads and writes.
//
// MPACT tells us the mnemonic and the raw 32-bit encoding. The register
// numbers come straight from the standard RISC-V fields (rd, rs1, rs2, rs3);
// which register file each field refers to depends on the opcode.

#pragma once

#include "Inst.hpp"

namespace coralnpu_perf {

// Fills inst.mnemonic, inst.cls, inst.srcs and inst.dsts from inst.disasm,
// inst.encoding and the vector configuration fields.
void decodeInst(Inst& inst);

}  // namespace coralnpu_perf
