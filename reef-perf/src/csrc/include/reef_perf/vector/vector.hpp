#pragma once

/** @file
 *  @brief Vector module: the RVV backend. Tree location: top.vector.
 *
 *  Units: top.vector.vxu (Vxu).
 *
 *  Receives vector instructions from the backend and owns their timing
 *  fields. Vector loads and stores are not sent here; they go to the
 *  backend's LSU (see exec_target()).
 */

#include "reef_perf/common/module.hpp"
#include "reef_perf/common/resource_pool.hpp"

#include <string>
#include <vector>

namespace reef_perf {

class Vxu;

/// The vector module.
class Vector : public Module {
  public:
    /// Public port: vector instructions in from the backend (InstPtr).
    static constexpr const char* kInInsts = "vxu.ports.in_insts";
    /// Public port: command-queue credits out to the backend.
    static constexpr const char* kOutCredits = "vxu.ports.out_credits";

    /// Creates the module; its tree node is top.vector.
    Vector();

    /** Registers Vxu's factory.
     *
     *  @param resources The simulation's resource set.
     */
    void add_factories(sparta::ResourceSet& resources) override;

    /// Looks up the vector unit.
    void bind() override;

    /** The module's pools.
     *
     *  @return {vector}.
     */
    [[nodiscard]] std::vector<const ResourcePool*> pools() const override;

  protected:
    /** The module's units.
     *
     *  @return {"vxu"}.
     */
    [[nodiscard]] std::vector<std::string> unit_names() const override;

  private:
    /// Vector unit, valid after bind().
    Vxu* vxu_ = nullptr;
};

} // namespace reef_perf
