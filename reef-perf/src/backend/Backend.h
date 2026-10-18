//===- Backend.h - The backend module ---------------------------*- C++ -*-===//
///
/// \file
/// The backend module (top.backend): dispatch, scalar execution, the LSU and
/// retirement. Vector and matrix instructions leave through OutVector and
/// OutMatrix; those modules own their timing.
///
//===----------------------------------------------------------------------===//

#ifndef REEF_PERF_BACKEND_BACKEND_H
#define REEF_PERF_BACKEND_BACKEND_H

#include "backend/Dispatch.h"
#include "common/Interfaces.h"
#include "common/Module.h"
#include "common/ResourcePool.h"

#include <cstdint>
#include <string>
#include <vector>

namespace reef_perf {

class Lsu;
class Rob;
class ScalarExec;

/// The backend module. Units: dispatch, scalar_exec, lsu, rob.
class Backend : public Module {
public:
  /// Public port: fetch groups in from the frontend (FetchPacket).
  static constexpr const char *InInsts = "dispatch.ports.in_insts";
  /// Public port: instruction-buffer credits out to the frontend.
  static constexpr const char *OutFetchCredits =
      "dispatch.ports.out_fetch_credits";
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

  /// Creates top.backend.
  Backend();

  /// Registers the factories of the backend's units in \p Resources.
  void addFactories(sparta::ResourceSet &Resources) override;

  /// Binds Dispatch, ScalarExec, the LSU and the Rob together.
  void bind() override;

  /// Connects the memory module \p Mem, which must outlive the simulation.
  void setMemory(MemoryInterface *Mem);

  /// \returns alu, mul, div, fpu, fdiv and lsu.
  std::vector<const ResourcePool *> getPools() const override;

  /// \returns the instructions retired.
  uint64_t getNumRetired() const;

  /// \returns the cycle of the last retirement: total cycles once the run
  /// ends.
  uint64_t getLastRetireCycle() const;

  /// \returns the cycles in which at least one instruction dispatched.
  uint64_t getCyclesWithDispatch() const;

  /// \returns the cycles in which nothing dispatched, blamed on \p Reason.
  uint64_t getStallCycles(StallReason Reason) const;

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

#endif // REEF_PERF_BACKEND_BACKEND_H
