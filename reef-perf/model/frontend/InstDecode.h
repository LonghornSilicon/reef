//===- InstDecode.h - Instruction classification ----------------*- C++ -*-===//
///
/// \file
/// Classifies an instruction and finds the registers it reads. Registers
/// *written* come from Spike's commit log (FuncSim); Spike does not log reads,
/// so sources are decoded from the RISC-V fields here.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_FRONTEND_INSTDECODE_H
#define REEF_PERF_FRONTEND_INSTDECODE_H

#include "common/Inst.h"

namespace reef_perf {

/// Sets the Mnemonic, Class and Srcs of \p I from its Disasm, Encoding and
/// vector configuration; leaves Dsts alone. x0 is never listed, since it is
/// never a dependency.
void decodeInst(Inst &I);

} // namespace reef_perf

#endif // REEF_PERF_FRONTEND_INSTDECODE_H
