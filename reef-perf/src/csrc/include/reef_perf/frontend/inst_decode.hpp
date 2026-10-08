#pragma once

/** @file
 *  @brief Classifies an instruction and finds the registers it reads.
 *
 *  Registers *written* come from Spike's commit log (SpikeDriver), which is
 *  exact. Spike does not log reads, so sources are decoded here from the
 *  standard RISC-V fields (rs1, rs2, rs3, vs1, vs2, vd); which register file
 *  a field reads depends on the opcode.
 */

#include "reef_perf/common/inst.hpp"

namespace reef_perf {

/** Fills in an instruction's mnemonic, class and source registers.
 *
 *  Reads inst.disasm, inst.encoding and the vector configuration fields
 *  (vl, sew_bytes, lmul8); writes inst.mnemonic, inst.cls and inst.srcs.
 *  inst.dsts is left alone. x0 is never listed, since it is never a
 *  dependency.
 *
 *  @param inst Instruction to decode in place.
 */
void decode_inst(Inst& inst);

} // namespace reef_perf
