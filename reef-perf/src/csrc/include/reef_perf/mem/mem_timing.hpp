#pragma once

/** @file
 *  @brief Timing models of the memories, as plain C++ classes so they can be
 *         tested without a Sparta simulation.
 *
 *  The Sparta units in tcm.hpp and axi.hpp read their parameters and own one
 *  of these models each.
 */

#include "reef_perf/common/inst.hpp"
#include "reef_perf/common/interfaces.hpp"
#include "reef_perf/common/resource_pool.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace reef_perf {

/** Distinct aligned blocks a set of accesses touches.
 *
 *  @param accesses The accesses; an access may straddle two blocks.
 *  @param block_bytes Block size in bytes; must be at least 1.
 *  @return Number of distinct blocks, at least 1 (an empty list counts as
 *          one block).
 */
std::uint32_t distinct_blocks(std::span<const MemAccess> accesses,
                              std::uint32_t block_bytes);

/** A tightly-coupled memory (ITCM or DTCM).
 *
 *  Super-coarse: one port that transfers one line every `cycles_per_line`
 *  cycles; latency = `latency` + transfer cycles - 1. No banks.
 */
class TcmModel {
  public:
    /** Creates a TCM.
     *
     *  @param name Pool name used in statistics, e.g. "dtcm".
     *  @param latency Latency of a single-line access.
     *  @param line_bytes Bytes per transfer; at least 1.
     *  @param cycles_per_line Cycles the port is busy per line.
     */
    TcmModel(std::string name, std::uint32_t latency,
             std::uint32_t line_bytes, std::uint32_t cycles_per_line);

    /** Times an access and books the port.
     *
     *  @param accesses The bytes accessed.
     *  @param earliest First cycle the access may start.
     *  @return Its timing.
     */
    MemResponse access(std::span<const MemAccess> accesses,
                       std::uint64_t earliest);

    /** The port, for the end-of-run summary.
     *
     *  @return The pool.
     */
    [[nodiscard]] const ResourcePool& port() const { return port_; }

  private:
    /// Latency of a single-line access.
    std::uint32_t latency_;
    /// Bytes per transfer.
    std::uint32_t line_bytes_;
    /// Cycles the port is busy per line.
    std::uint32_t cycles_per_line_;
    /// The memory port.
    ResourcePool port_;
};

/** The AXI master port and the memory behind it, as a black box.
 *
 *  Super-coarse: the data channel moves one beat per cycle; at most
 *  `max_outstanding` operations are in flight; every operation sees a fixed
 *  `latency` from the black box. Read and write channels are not separated.
 */
class AxiModel {
  public:
    /** Creates the AXI port.
     *
     *  @param latency Round-trip latency of a one-beat operation.
     *  @param bytes_per_beat Data bus width in bytes; at least 1.
     *  @param max_outstanding Operations in flight at once; at least 1.
     */
    AxiModel(std::uint32_t latency, std::uint32_t bytes_per_beat,
             std::uint32_t max_outstanding);

    /** Times an operation and books the data channel and an outstanding
     *  slot.
     *
     *  @param accesses The bytes accessed.
     *  @param earliest First cycle the operation may start.
     *  @return Its timing.
     */
    MemResponse access(std::span<const MemAccess> accesses,
                       std::uint64_t earliest);

    /** The pools, for the end-of-run summary.
     *
     *  @return {axi_data, axi_outstanding}.
     */
    [[nodiscard]] std::vector<const ResourcePool*> pools() const {
        return {&data_, &outstanding_};
    }

  private:
    /// Round-trip latency of a one-beat operation.
    std::uint32_t latency_;
    /// Data bus width in bytes.
    std::uint32_t bytes_per_beat_;
    /// The data channel: one beat per cycle.
    ResourcePool data_;
    /// Outstanding-transaction slots, each held until its operation is done.
    ResourcePool outstanding_;
};

} // namespace reef_perf
