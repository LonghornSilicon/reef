//===- reef_trace.cpp - Spike instruction-stream dump ---------------------===//
///
/// \file
/// Prints the dynamic instruction stream of an ELF as Spike executes it.
///
///   reef_trace <elf> [max_instructions]
///
/// One line per instruction: sequence number, pc, encoding, next pc,
/// vl/SEW/LMUL, decoded class, number of memory accesses and disassembly.
/// Useful for checking a workload before running the timing model.
///
//===----------------------------------------------------------------------===//

#include "common/Inst.h"
#include "frontend/FuncSim.h"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <string>

using namespace reef_perf;

/// Prints the trace. \returns the exit code.
static int run(int Argc, char **Argv) {
  if (Argc < 2) {
    std::fprintf(stderr, "usage: %s <elf> [max_instructions]\n", Argv[0]);
    return 2;
  }
  uint64_t MaxInsts = Argc > 2 ? std::stoull(Argv[2], nullptr, 0)
                               : std::numeric_limits<uint64_t>::max();

  FuncSim Sim(Argv[1], SpikeOptions{});
  uint64_t Count = 0;
  while (Count < MaxInsts) {
    InstRef I = Sim.next();
    if (!I)
      break;
    std::printf("%6llu %08x %08x -> %08x vl=%u e%u m%u/8 %-11s mem=%zu  %s\n",
                static_cast<unsigned long long>(I->Seq), I->PC, I->Encoding,
                I->NextPC, I->VL, I->SEWBytes * 8U, I->LMUL8,
                getClassName(I->Class), I->Mem.size(), I->Disasm.c_str());
    ++Count;
  }
  std::printf("# %llu instructions, %s\n",
              static_cast<unsigned long long>(Count),
              Sim.isHalted() ? "halted" : "limit reached");
  return 0;
}

int main(int argc, char **argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception &E) {
    std::fprintf(stderr, "reef_trace: %s\n", E.what());
    return 1;
  }
}
