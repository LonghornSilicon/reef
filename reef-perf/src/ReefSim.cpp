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
  return {&FrontendModule, &BackendModule, &VectorModule, &MatrixModule,
          &MemModule};
}

std::vector<const Module *> ReefSim::getModules() const {
  return {&FrontendModule, &BackendModule, &VectorModule, &MatrixModule,
          &MemModule};
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
  bindPorts(FrontendModule, Fetch::OutInsts, BackendModule, Backend::InInsts);
  bindPorts(BackendModule, Backend::OutFetchCredits, FrontendModule,
            Fetch::InCredits);
  bindPorts(BackendModule, Backend::OutVector, VectorModule, Vector::InInsts);
  bindPorts(VectorModule, Vector::OutCredits, BackendModule,
            Backend::InVectorCredits);
  bindPorts(BackendModule, Backend::OutMatrix, MatrixModule, Matrix::InInsts);
  bindPorts(MatrixModule, Matrix::OutCredits, BackendModule,
            Backend::InMatrixCredits);

  FrontendModule.getUnit()->setFuncSim(TheFuncSim.get());
  MemModule.getUnit()->setMemoryMap(Regions);
  BackendModule.setMemory(MemModule.getUnit());
}

void ReefSim::printSummary(std::ostream &OS) const {
  uint64_t Cycles = BackendModule.getLastRetireCycle();
  uint64_t Insts = BackendModule.getNumRetired();
  OS << "\n==================== reef_perf summary ====================\n"
     << "  workload            : " << ElfPath << "\n"
     << "  instructions retired: " << Insts << "\n"
     << "  cycles              : " << Cycles << "\n"
     << "  IPC                 : " << std::fixed << std::setprecision(3)
     << getIPC(Insts, Cycles) << "\n"
     << "  taken branches/jumps: "
     << FrontendModule.getUnit()->getNumRedirects() << "\n";
  if (Insts != TheFuncSim->getNumExecuted())
    OS << "  WARNING: retired " << Insts << " but Spike executed "
       << TheFuncSim->getNumExecuted() << " instructions (model deadlock?)\n";

  OS << "\n  Dispatch: cycles with no dispatch, by cause\n";
  OS << std::setprecision(1);
  uint64_t Busy = BackendModule.getCyclesWithDispatch();
  OS << "    " << std::left << std::setw(20) << "(dispatched)" << std::right
     << std::setw(12) << Busy << "  " << std::setw(5)
     << getPercent(Busy, Cycles) << "%\n";
  constexpr auto NumReasons = static_cast<size_t>(StallReason::NumReasons);
  for (size_t Idx = 0; Idx < NumReasons; ++Idx) {
    auto Reason = static_cast<StallReason>(Idx);
    uint64_t Stalled = BackendModule.getStallCycles(Reason);
    OS << "    " << std::left << std::setw(20) << getStallReasonName(Reason)
       << std::right << std::setw(12) << Stalled << "  " << std::setw(5)
       << getPercent(Stalled, Cycles) << "%\n";
  }

  OS << "\n  Pools: utilization (busy cycles / (units * cycles))\n";
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
  uint64_t Cycles = BackendModule.getLastRetireCycle();
  uint64_t Insts = BackendModule.getNumRetired();
  OS << "{\n"
     << R"(  "workload": ")" << ElfPath << "\",\n"
     << "  \"instructions\": " << Insts << ",\n"
     << "  \"functional_instructions\": " << TheFuncSim->getNumExecuted()
     << ",\n"
     << "  \"cycles\": " << Cycles << ",\n"
     << "  \"ipc\": " << getIPC(Insts, Cycles) << ",\n"
     << "  \"taken_redirects\": " << FrontendModule.getUnit()->getNumRedirects()
     << ",\n"
     << "  \"dispatch\": {\n"
     << "    \"cycles_with_dispatch\": "
     << BackendModule.getCyclesWithDispatch();
  constexpr auto NumReasons = static_cast<size_t>(StallReason::NumReasons);
  for (size_t Idx = 0; Idx < NumReasons; ++Idx) {
    auto Reason = static_cast<StallReason>(Idx);
    OS << ",\n    \"stall_" << getStallReasonName(Reason)
       << "\": " << BackendModule.getStallCycles(Reason);
  }
  // Pool names are unique across modules (docs/interfaces.md), so the JSON
  // keeps its flat {"alu": ..., "vector": ...} layout.
  OS << "\n  },\n  \"pools\": {";
  bool First = true;
  for (const Module *M : getModules()) {
    for (const ResourcePool *Pool : M->getPools()) {
      OS << (First ? "\n" : ",\n") << "    \"" << Pool->getName()
         << R"(": {"units": )" << Pool->getCount()
         << ", \"ops\": " << Pool->getOps()
         << ", \"busy_cycles\": " << Pool->getBusyCycles() << "}";
      First = false;
    }
  }
  OS << "\n  }\n}\n";
}
