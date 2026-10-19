//===- ReefSim.h - The top-level simulation ---------------------*- C++ -*-===//
///
/// \file
/// The top-level Sparta simulation: builds the modules, binds the ports
/// between them and reports the results. It only knows each module's public
/// ports and summary accessors; see docs/interfaces.md.
///
/// Tree (parameters are top.MODULE[.UNIT].params.NAME):
///   top.frontend  fetch, with Spike and decode behind it
///   top.core      dispatch, scalar_exec, lsu, rob
///   top.vector    the RVV backend
///   top.matrix    the matrix engine (stub)
///   top.memory    ITCM, DTCM and the AXI port
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_REEFSIM_H
#define REEF_PERF_REEFSIM_H

#include "common/MemoryMap.h"
#include "common/Module.h"
#include "core/Core.h"
#include "frontend/Fetch.h"
#include "frontend/FuncSim.h"
#include "matrix/Matrix.h"
#include "memory/Memory.h"
#include "vector/Vector.h"

#include "sparta/app/Simulation.hpp"

#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace reef_perf {

/// The Reef simulation.
class ReefSim : public sparta::app::Simulation {
public:
  /// Creates a simulation driven by \p Scheduler that runs the RISC-V ELF
  /// \p ElfPath on a machine configured by \p Options.
  /// \throws std::runtime_error if the ELF cannot be loaded.
  ReefSim(sparta::Scheduler &Scheduler, const std::string &ElfPath,
          const SpikeOptions &Options);
  ~ReefSim() override; ///< Tears down the tree.

  /// Writes the human-readable end-of-run summary to \p OS.
  void printSummary(std::ostream &OS) const;

  /// Writes the end-of-run summary to \p OS as JSON.
  void writeJSON(std::ostream &OS) const;

private:
  // Sparta's virtual hooks keep Sparta's names.
  void buildTree_() override;
  void configureTree_() override;
  void bindTree_() override;

  std::vector<Module *> getModules();
  std::vector<const Module *> getModules() const;
  void bindPorts(const Module &From, const char *FromPort, const Module &To,
                 const char *ToPort);

  std::unique_ptr<FuncSim> TheFuncSim;
  std::string ElfPath;
  std::vector<MemoryRegion> Regions;

  UnitModule<Fetch, Fetch::FetchParameterSet> FrontendModule{
      "frontend", "Fetch, with Spike and decode behind it"};
  Core CoreModule;
  UnitModule<Vector, Vector::VectorParameterSet> VectorModule{"vector",
                                                              "RVV backend"};
  UnitModule<Matrix, Matrix::MatrixParameterSet> MatrixModule{"matrix",
                                                              "Matrix engine"};
  UnitModule<Memory, Memory::MemoryParameterSet> MemoryModule{
      "memory", "TCMs and the AXI port"};
};

} // namespace reef_perf

#endif // REEF_PERF_REEFSIM_H
