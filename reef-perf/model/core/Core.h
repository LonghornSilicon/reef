//===- Core.h - The core module (scalar pipeline) ---------------*- C++ -*-===//
///
/// \file
/// The core module (top.core): dispatch, scalar execution, the LSU and
/// retirement. Vector and matrix instructions leave through OutVector and
/// OutMatrix; those modules own their timing. The vector module books the LSU
/// for vector loads and stores through getVectorLsu().
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_CORE_CORE_H
#define REEF_PERF_CORE_CORE_H

#include "common/Interfaces.h"
#include "common/Module.h"
#include "common/ResourcePool.h"
#include "core/Dispatch.h"
#include "core/Rob.h"

#include <cstdint>
#include <string>
#include <vector>

namespace reef_perf {

class Lsu;
class ScalarExec;

/// The core module. Units: dispatch, scalar_exec, lsu, rob.
class Core : public Module {
public:
  /// Public port: vector instructions out (InstRef).
  static constexpr const char *OutVector = "dispatch.ports.out_vector";
  /// Public port: vector command-queue credits in.
  static constexpr const char *InVectorCredits =
      "dispatch.ports.in_vec_credits";
  /// Public port: matrix instructions out (InstRef).
  static constexpr const char *OutMatrix = "dispatch.ports.out_matrix";
  /// Public port: matrix command-queue credits in.
  static constexpr const char *InMatrixCredits =
      "dispatch.ports.in_mtx_credits";

  /// Creates top.core.
  Core();

  /// Registers the factories of the core's units in \p Resources.
  void addFactories(sparta::ResourceSet &Resources) override;

  /// Binds Dispatch, ScalarExec, the LSU and the Rob together.
  void bind() override;

  /// Connects the memory module \p Mem, which must outlive the simulation.
  void setMemory(MemoryInterface *Mem);

  /// \returns the LSU as the vector module books it for vector loads and
  /// stores. Valid after bind().
  VectorLsuInterface *getVectorLsu() const;

  /// Connects the frontend's instruction buffer \p Buffer, which must outlive
  /// the simulation.
  void setInstBuffer(InstBufferInterface *Buffer);

  /// \returns alu, mul, div, fpu, fdiv and lsu.
  std::vector<const ResourcePool *> getPools() const override;

  /// \returns the instructions retired.
  uint64_t getNumRetired() const;

  /// \returns the cycle of the last retirement: total cycles once the run
  /// ends.
  Cycle getLastRetireCycle() const;

  /// \returns the cycles in which at least one instruction dispatched.
  uint64_t getCyclesWithDispatch() const;

  /// \returns the cycles in which nothing dispatched, blamed on \p Reason.
  uint64_t getStallCycles(StallReason Reason) const;

  /// \returns where the dispatch slots went (top-down analysis).
  const TopDownSlots &getTopDownSlots() const;

  /// \returns every retired read of mcycle, in program order.
  const std::vector<McycleRead> &getMcycleReads() const;

protected:
  /// \returns {"dispatch", "scalar_exec", "lsu", "rob"}.
  std::vector<std::string> getUnitNames() const override;

private:
  Dispatch *TheDispatch = nullptr;
  ScalarExec *TheScalarExec = nullptr;
  Lsu *TheLsu = nullptr;
  Rob *TheRob = nullptr;
};

} // namespace reef_perf

#endif // REEF_PERF_CORE_CORE_H
