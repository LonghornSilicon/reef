#pragma once

/** @file
 *  @brief The AXI master port to off-core memory. Tree location:
 *         top.mem.axi.
 */

#include "reef_perf/common/interfaces.hpp"
#include "reef_perf/common/resource_pool.hpp"
#include "reef_perf/mem/mem_timing.hpp"

#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/Unit.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace reef_perf {

/// The AXI master port and the memory behind it. Tree location: top.mem.axi.
class Axi : public sparta::Unit {
  public:
    /// Parameters of the AXI port (top.mem.axi.params).
    class AxiParameterSet : public sparta::ParameterSet {
      public:
        /** Registers the parameters with the tree node.
         *
         *  @param node The parameter set's tree node.
         */
        explicit AxiParameterSet(sparta::TreeNode* node)
            : sparta::ParameterSet(node) {}

        /// Round-trip latency of a one-beat operation (black box).
        PARAMETER(std::uint32_t, axi_latency, 20,
                  "Round-trip latency of a one-beat AXI operation")
        /// Data bus width in bytes.
        PARAMETER(std::uint32_t, axi_bytes_per_beat, 32,
                  "AXI data bus width in bytes")
        /// Operations in flight at once.
        PARAMETER(std::uint32_t, axi_max_outstanding, 4,
                  "AXI operations in flight at once")
    };

    /// Name of this unit in the Sparta tree. Sparta's ResourceFactory
    /// requires a static member called exactly `name`.
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr const char* name = "axi";

    /** Creates the unit.
     *
     *  @param node Tree node the unit is attached to.
     *  @param params The unit's parameters.
     */
    Axi(sparta::TreeNode* node, const AxiParameterSet* params);

    /** Times an operation and books the port.
     *
     *  @param accesses The bytes accessed.
     *  @param earliest First cycle the operation may start.
     *  @return Its timing.
     */
    MemResponse access(std::span<const MemAccess> accesses,
                       std::uint64_t earliest) {
        return model_.access(accesses, earliest);
    }

    /** The pools, for the end-of-run summary.
     *
     *  @return {axi_data, axi_outstanding}.
     */
    [[nodiscard]] std::vector<const ResourcePool*> pools() const {
        return model_.pools();
    }

  private:
    /// The timing model.
    AxiModel model_;
};

} // namespace reef_perf
