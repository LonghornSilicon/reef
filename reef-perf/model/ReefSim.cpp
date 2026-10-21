//===- ReefSim.cpp - The top-level simulation -----------------------------===//

#include "ReefSim.h"
#include "common/ResourcePool.h"

#include "sparta/ports/Port.hpp"

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

using namespace reef_perf;

/// \returns 100 * \p Part / \p Whole, or 0 when \p Whole is 0.
static double getPercent(uint64_t Part, uint64_t Whole) {
  return Whole == 0
             ? 0.0
             : 100.0 * static_cast<double>(Part) / static_cast<double>(Whole);
}

/// \returns \p Insts / \p Cycles, or 0 when no cycles elapsed.
static double getIPC(uint64_t Insts, uint64_t Cycles) {
  return Cycles == 0 ? 0.0
                     : static_cast<double>(Insts) / static_cast<double>(Cycles);
}

/// Writes \p Slots as the "topdown" JSON object, followed by a comma.
/// Backend reasons that never occurred are left out.
static void writeTopDown(std::ostream &OS, const TopDownSlots &Slots) {
  OS << "  \"topdown\": {\n"
     << "    \"dispatch_width\": " << Slots.Width << ",\n"
     << "    \"slots\": " << Slots.getTotal() << ",\n"
     << "    \"retiring\": " << Slots.Retiring << ",\n"
     << "    \"frontend_bound\": {";
  for (size_t Idx = 0; Idx < TopDownSlots::NumFrontend; ++Idx)
    OS << (Idx ? ", " : "") << '"'
       << getFrontendStallName(static_cast<FrontendStall>(Idx))
       << "\": " << Slots.Frontend[Idx];
  OS << "},\n    \"backend_bound\": {";
  for (size_t Target = 0; Target < NumExecTargets; ++Target) {
    OS << (Target ? ",\n      " : "\n      ") << '"'
       << getBoundName(static_cast<ExecTarget>(Target)) << "\": {";
    bool First = true;
    for (size_t Reason = 0; Reason < TopDownSlots::NumReasons; ++Reason) {
      uint64_t N = Slots.Backend[Target][Reason];
      if (N == 0)
        continue;
      OS << (First ? "" : ", ") << '"'
         << getStallReasonName(static_cast<StallReason>(Reason)) << "\": " << N;
      First = false;
    }
    OS << "}";
  }
  OS << "\n    }\n  },\n";
}

ReefSim::ReefSim(sparta::Scheduler &Scheduler, const std::string &ElfPath,
                 const SpikeOptions &Options)
    : sparta::app::Simulation("reef_perf", &Scheduler),
      TheFuncSim(std::make_unique<FuncSim>(ElfPath, Options)), ElfPath(ElfPath),
      Regions(Options.Regions) {
  for (Module *M : getModules())
    M->addFactories(*getResourceSet());
}

ReefSim::~ReefSim() { getRoot()->enterTeardown(); }

std::vector<Module *> ReefSim::getModules() {
  return {&FrontendModule, &CoreModule, &VectorModule, &MatrixModule,
          &MemoryModule};
}

std::vector<const Module *> ReefSim::getModules() const {
  return {&FrontendModule, &CoreModule, &VectorModule, &MatrixModule,
          &MemoryModule};
}

void ReefSim::buildTree_() {
  for (Module *M : getModules())
    M->build(getRoot(), *getResourceSet(), to_delete_);
}

void ReefSim::configureTree_() {}

void ReefSim::bindPorts(const Module &From, const char *FromPort,
                        const Module &To, const char *ToPort) {
  sparta::TreeNode *Root = getRoot();
  sparta::bind(Root->getChildAs<sparta::Port>(From.getPortPath(FromPort)),
               Root->getChildAs<sparta::Port>(To.getPortPath(ToPort)));
}

void ReefSim::bindTree_() {
  for (Module *M : getModules())
    M->bind();

  // The interfaces between modules (docs/interfaces.md).
  bindPorts(CoreModule, Core::OutVector, VectorModule, Vector::InInsts);
  bindPorts(VectorModule, Vector::OutCredits, CoreModule,
            Core::InVectorCredits);
  bindPorts(CoreModule, Core::OutMatrix, MatrixModule, Matrix::InInsts);
  bindPorts(MatrixModule, Matrix::OutCredits, CoreModule,
            Core::InMatrixCredits);

  FrontendModule.getUnit()->setFuncSim(TheFuncSim.get());
  MemoryModule.getUnit()->setMemoryMap(Regions);
  CoreModule.setMemory(MemoryModule.getUnit());
  CoreModule.setInstBuffer(FrontendModule.getUnit());
}

void ReefSim::printSummary(std::ostream &OS) const {
  uint64_t Cycles = CoreModule.getLastRetireCycle();
  uint64_t Insts = CoreModule.getNumRetired();
  OS << "\n==================== reef_perf summary ====================\n"
     << "  program             : " << ElfPath << "\n"
     << "  instructions retired: " << Insts << "\n"
     << "  cycles              : " << Cycles << "\n"
     << "  IPC                 : " << std::fixed << std::setprecision(3)
     << getIPC(Insts, Cycles) << "\n"
     << "  taken branches/jumps: "
     << FrontendModule.getUnit()->getNumRedirects() << "\n";
  const std::vector<McycleRead> &Reads = CoreModule.getMcycleReads();
  if (Reads.size() >= 2)
    OS << "  ROI cycles (mcycle) : "
       << Reads.back().IssueCycle - Reads.front().IssueCycle << "\n";
  if (Insts != TheFuncSim->getNumExecuted())
    OS << "  WARNING: retired " << Insts << " but Spike executed "
       << TheFuncSim->getNumExecuted() << " instructions (model deadlock?)\n";

  OS << "\n  Dispatch: cycles with no dispatch, by cause\n";
  OS << std::setprecision(1);
  uint64_t Busy = CoreModule.getCyclesWithDispatch();
  OS << "    " << std::left << std::setw(20) << "(dispatched)" << std::right
     << std::setw(12) << Busy << "  " << std::setw(5)
     << getPercent(Busy, Cycles) << "%\n";
  constexpr auto NumReasons = static_cast<size_t>(StallReason::NumReasons);
  for (size_t Idx = 0; Idx < NumReasons; ++Idx) {
    auto Reason = static_cast<StallReason>(Idx);
    uint64_t Stalled = CoreModule.getStallCycles(Reason);
    OS << "    " << std::left << std::setw(20) << getStallReasonName(Reason)
       << std::right << std::setw(12) << Stalled << "  " << std::setw(5)
       << getPercent(Stalled, Cycles) << "%\n";
  }

  // Top-down, levels 1 and 2 (docs/topdown.md).
  const TopDownSlots &Slots = CoreModule.getTopDownSlots();
  uint64_t Total = Slots.getTotal();
  auto PrintSlots = [&](const std::string &Label, uint64_t N) {
    OS << "    " << std::left << std::setw(20) << Label << std::right
       << std::setw(12) << N << "  " << std::setw(5) << getPercent(N, Total)
       << "%\n";
  };
  OS << "\n  Top-down: dispatch slots (" << Slots.Width
     << " per cycle; docs/topdown.md)\n";
  PrintSlots("retiring", Slots.Retiring);
  uint64_t FrontendBound = 0;
  for (uint64_t N : Slots.Frontend)
    FrontendBound += N;
  PrintSlots("frontend_bound", FrontendBound);
  for (size_t Idx = 0; Idx < TopDownSlots::NumFrontend; ++Idx)
    PrintSlots(std::string("  ") +
                   getFrontendStallName(static_cast<FrontendStall>(Idx)),
               Slots.Frontend[Idx]);
  PrintSlots("backend_bound", Total - Slots.Retiring - FrontendBound);
  for (size_t Target = 0; Target < NumExecTargets; ++Target) {
    uint64_t Bound = 0;
    for (uint64_t N : Slots.Backend[Target])
      Bound += N;
    PrintSlots(std::string("  ") +
                   getBoundName(static_cast<ExecTarget>(Target)),
               Bound);
  }

  OS << "\n  Pools: utilization (busy cycles / (resources * cycles))\n";
  for (const Module *M : getModules())
    for (const ResourcePool *Pool : M->getPools())
      OS << "    " << std::left << std::setw(16)
         << (M->getName() + "." + Pool->getName()) << std::right << " x"
         << Pool->getCount() << "  ops " << std::setw(10) << Pool->getOps()
         << "  util " << std::setw(5)
         << getPercent(Pool->getBusyCycles(), Pool->getCount() * Cycles)
         << "%\n";
  OS << "============================================================\n";
}

void ReefSim::writeJSON(std::ostream &OS) const {
  uint64_t Cycles = CoreModule.getLastRetireCycle();
  uint64_t Insts = CoreModule.getNumRetired();
  OS << "{\n"
     << R"(  "program": ")" << ElfPath << "\",\n"
     << "  \"instructions\": " << Insts << ",\n"
     << "  \"functional_instructions\": " << TheFuncSim->getNumExecuted()
     << ",\n"
     << "  \"cycles\": " << Cycles << ",\n"
     << "  \"ipc\": " << getIPC(Insts, Cycles) << ",\n"
     << "  \"taken_redirects\": " << FrontendModule.getUnit()->getNumRedirects()
     << ",\n";
  // The microbenchmarks read mcycle at the start and end of their region of
  // interest; the difference is what the RTL measures.
  // ROI instructions exclude the two reads, as the RTL trace parser does.
  const std::vector<McycleRead> &Reads = CoreModule.getMcycleReads();
  OS << "  \"mcycle_reads\": [";
  for (size_t Idx = 0; Idx < Reads.size(); ++Idx)
    OS << (Idx ? ", " : "") << Reads[Idx].IssueCycle;
  OS << "],\n";
  if (Reads.size() >= 2)
    OS << "  \"roi_cycles\": "
       << Reads.back().IssueCycle - Reads.front().IssueCycle << ",\n"
       << "  \"roi_instructions\": " << Reads.back().Seq - Reads.front().Seq - 1
       << ",\n";
  else
    OS << "  \"roi_cycles\": null,\n  \"roi_instructions\": null,\n";
  OS << "  \"dispatch\": {\n"
     << "    \"cycles_with_dispatch\": " << CoreModule.getCyclesWithDispatch();
  constexpr auto NumReasons = static_cast<size_t>(StallReason::NumReasons);
  for (size_t Idx = 0; Idx < NumReasons; ++Idx) {
    auto Reason = static_cast<StallReason>(Idx);
    OS << ",\n    \"stall_" << getStallReasonName(Reason)
       << "\": " << CoreModule.getStallCycles(Reason);
  }
  // Pool names are unique across modules (docs/interfaces.md), so the JSON
  // keeps its flat {"alu": ..., "vector": ...} layout.
  OS << "\n  },\n";
  writeTopDown(OS, CoreModule.getTopDownSlots());
  OS << "  \"pools\": {";
  bool First = true;
  for (const Module *M : getModules()) {
    for (const ResourcePool *Pool : M->getPools()) {
      OS << (First ? "\n" : ",\n") << "    \"" << Pool->getName()
         << R"(": {"count": )" << Pool->getCount()
         << ", \"ops\": " << Pool->getOps()
         << ", \"busy_cycles\": " << Pool->getBusyCycles() << "}";
      First = false;
    }
  }
  OS << "\n  }\n}\n";
}
