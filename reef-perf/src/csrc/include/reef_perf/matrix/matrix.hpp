#pragma once

/** @file
 *  @brief Matrix module: the matrix engine. Tree location: top.matrix.
 *
 *  Units: top.matrix.mxu (Mxu, a stub).
 *
 *  Receives matrix instructions from the backend through the same kind of
 *  interface as the vector module (instructions in, queue credits out) and
 *  owns their timing fields. The matrix ISA is not defined yet, so nothing
 *  decodes to InstClass::MATRIX and the module sees no traffic.
 */

#include "reef_perf/common/module.hpp"
#include "reef_perf/common/resource_pool.hpp"

#include <string>
#include <vector>

namespace reef_perf {

class Mxu;

/// The matrix module.
class Matrix : public Module {
  public:
    /// Public port: matrix instructions in from the backend (InstPtr).
    static constexpr const char* kInInsts = "mxu.ports.in_insts";
    /// Public port: command-queue credits out to the backend.
    static constexpr const char* kOutCredits = "mxu.ports.out_credits";

    /// Creates the module; its tree node is top.matrix.
    Matrix();

    /** Registers Mxu's factory.
     *
     *  @param resources The simulation's resource set.
     */
    void add_factories(sparta::ResourceSet& resources) override;

    /// Looks up the matrix unit.
    void bind() override;

    /** The module's pools.
     *
     *  @return {matrix}.
     */
    [[nodiscard]] std::vector<const ResourcePool*> pools() const override;

  protected:
    /** The module's units.
     *
     *  @return {"mxu"}.
     */
    [[nodiscard]] std::vector<std::string> unit_names() const override;

  private:
    /// Matrix unit, valid after bind().
    Mxu* mxu_ = nullptr;
};

} // namespace reef_perf
