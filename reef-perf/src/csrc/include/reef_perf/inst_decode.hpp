#pragma once

/** @file
 *  @brief Classifies an instruction and finds the registers it uses.
 *
 *  The functional simulator gives us the disassembly and the raw 32-bit
 *  encoding. Register numbers come straight from the standard RISC-V fields
 *  (rd, rs1, rs2, rs3); which register file each field refers to depends on
 *  the opcode.
 */

#include "reef_perf/inst.hpp"

namespace reef_perf {

/** Fills in an instruction's mnemonic, class and register operands.
 *
 *  Reads inst.disasm, inst.encoding and the vector configuration fields
 *  (vl, sew_bytes, lmul8); writes inst.mnemonic, inst.cls, inst.srcs and
 *  inst.dsts. x0 is never listed, since it is never a dependency.
 *
 *  @param inst Instruction to decode in place.
 */
void decode_inst(Inst& inst);

} // namespace reef_perf
