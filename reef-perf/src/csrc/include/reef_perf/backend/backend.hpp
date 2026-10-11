#pragma once

/** @file
 *  @brief Backend module: dispatch, scalar execution and retirement.
 *         Tree location: top.backend.
 *
 *  Units:
 *  - top.backend.dispatch     Dispatch (instruction buffer, scoreboard,
 *                             credits, routing)
 *  - top.backend.scalar_exec  ScalarExec (integer and FP units)
 *  - top.backend.lsu          Lsu (scalar and vector loads and stores)
 *  - top.backend.rob          Rob
 *
 *  Vector and matrix instructions leave the backend through kOutVector and
 *  kOutMatrix; those modules own their timing.
 */

#include "reef_perf/backend/dispatch.hpp"
#include "reef_perf/common/interfaces.hpp"
#include "reef_perf/common/module.hpp"
#include "reef_perf/common/resource_pool.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace reef_perf {

class Lsu;
class Rob;
class ScalarExec;

/// The backend module.
class Backend : public Module {
  public:
    /// Public port: fetch groups in from the frontend (FetchPacket).
    static constexpr const char* kInInsts = "dispatch.ports.in_insts";
    /// Public port: instruction-buffer credits out to the frontend.
    static constexpr const char* kOutFetchCredits =
        "dispatch.ports.out_fetch_credits";
    /// Public port: vector instructions out to the vector module (InstPtr).
    static constexpr const char* kOutVector = "dispatch.ports.out_vector";
    /// Public port: vector command-queue credits in from the vector module.
    static constexpr const char* kInVectorCredits =
        "dispatch.ports.in_vec_credits";
    /// Public port: matrix instructions out to the matrix module (InstPtr).
    static constexpr const char* kOutMatrix = "dispatch.ports.out_matrix";
    /// Public port: matrix command-queue credits in from the matrix module.
    static constexpr const char* kInMatrixCredits =
        "dispatch.ports.in_mtx_credits";

    /// Creates the module; its tree node is top.backend.
    Backend();

    /** Registers the factories of the backend's units.
     *
     *  @param resources The simulation's resource set.
     */
    void add_factories(sparta::ResourceSet& resources) override;

    /// Binds Dispatch, the scalar units, the LSU and the Rob together.
    void bind() override;

    /** Connects the memory module to the LSU. Call after bind().
     *
     *  @param memory The memory module; must outlive the simulation.
     */
    void set_memory(MemoryInterface* memory);

    /** The backend's pools.
     *
     *  @return alu, mul, div, fpu, fdiv, lsu.
     */
    [[nodiscard]] std::vector<const ResourcePool*> pools() const override;

    /** Instructions retired.
     *
     *  @return Instruction count.
     */
    [[nodiscard]] std::uint64_t num_retired() const;

    /** Cycle of the last retirement: total cycles once the run is over.
     *
     *  @return Cycle number.
     */
    [[nodiscard]] std::uint64_t last_retire_cycle() const;

    /** Cycles in which at least one instruction dispatched.
     *
     *  @return Cycle count.
     */
    [[nodiscard]] std::uint64_t cycles_with_dispatch() const;

    /** Cycles in which nothing dispatched, blamed on one reason.
     *
     *  @param reason The stall reason.
     *  @return Cycle count.
     */
    [[nodiscard]] std::uint64_t stall_cycles(StallReason reason) const;

  protected:
    /** The module's units.
     *
     *  @return {"dispatch", "scalar_exec", "lsu", "rob"}.
     */
    [[nodiscard]] std::vector<std::string> unit_names() const override;

  private:
    /// Dispatch unit, valid after bind().
    Dispatch* dispatch_ = nullptr;
    /// Scalar units, valid after bind().
    ScalarExec* scalar_exec_ = nullptr;
    /// Load/store unit, valid after bind().
    Lsu* lsu_ = nullptr;
    /// Retirement buffer, valid after bind().
    Rob* rob_ = nullptr;
};

} // namespace reef_perf
