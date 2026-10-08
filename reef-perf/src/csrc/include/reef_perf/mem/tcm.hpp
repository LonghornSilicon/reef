#pragma once

/** @file
 *  @brief The ITCM and DTCM. Tree location: top.mem.tcm.
 */

#include "reef_perf/common/interfaces.hpp"
#include "reef_perf/common/memory_map.hpp"
#include "reef_perf/common/resource_pool.hpp"
#include "reef_perf/mem/mem_timing.hpp"

#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/Unit.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace reef_perf {

/// The tightly-coupled memories. Tree location: top.mem.tcm.
class Tcm : public sparta::Unit {
  public:
    /// Parameters of the TCMs (top.mem.tcm.params).
    class TcmParameterSet : public sparta::ParameterSet {
      public:
        /** Registers the parameters with the tree node.
         *
         *  @param node The parameter set's tree node.
         */
        explicit TcmParameterSet(sparta::TreeNode* node)
            : sparta::ParameterSet(node) {}

        /// ITCM latency of a single-line access.
        PARAMETER(std::uint32_t, itcm_latency, 2,
                  "ITCM latency of a single-line access")
        /// ITCM bytes per transfer.
        PARAMETER(std::uint32_t, itcm_line_bytes, 16, "ITCM bytes per transfer")
        /// Cycles the ITCM port is busy per line.
        PARAMETER(std::uint32_t, itcm_cycles_per_line, 1,
                  "Cycles the ITCM port is busy per line")
        /// DTCM latency of a single-line access (load-to-use).
        PARAMETER(std::uint32_t, dtcm_latency, 2,
                  "DTCM latency of a single-line access")
        /// DTCM bytes per transfer.
        PARAMETER(std::uint32_t, dtcm_line_bytes, 16, "DTCM bytes per transfer")
        /// Cycles the DTCM port is busy per line.
        PARAMETER(std::uint32_t, dtcm_cycles_per_line, 1,
                  "Cycles the DTCM port is busy per line")
    };

    /// Name of this unit in the Sparta tree. Sparta's ResourceFactory
    /// requires a static member called exactly `name`.
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr const char* name = "tcm";

    /** Creates the unit.
     *
     *  @param node Tree node the unit is attached to.
     *  @param params The unit's parameters.
     */
    Tcm(sparta::TreeNode* node, const TcmParameterSet* params);

    /** Times an access to one of the TCMs and books its port.
     *
     *  @param kind RegionKind::ITCM or RegionKind::DTCM.
     *  @param accesses The bytes accessed.
     *  @param earliest First cycle the access may start.
     *  @return Its timing.
     */
    MemResponse access(RegionKind kind, std::span<const MemAccess> accesses,
                       std::uint64_t earliest);

    /** The pools, for the end-of-run summary.
     *
     *  @return {itcm, dtcm}.
     */
    [[nodiscard]] std::vector<const ResourcePool*> pools() const {
        return {&itcm_.port(), &dtcm_.port()};
    }

  private:
    /// The ITCM.
    TcmModel itcm_;
    /// The DTCM.
    TcmModel dtcm_;
};

} // namespace reef_perf
