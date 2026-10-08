#pragma once

/** @file
 *  @brief A group of identical execution resources.
 */

#include <cstdint>
#include <string>
#include <vector>

namespace reef_perf {

/** A pool of identical units, such as the four ALUs or the single LSU.
 *
 *  An operation books one unit for a number of cycles (its occupancy; 1 means
 *  fully pipelined). If every unit is busy, the operation starts when the
 *  earliest one frees up.
 */
class ResourcePool {
  public:
    /** Creates a pool.
     *
     *  @param name Name used in statistics, e.g. "alu".
     *  @param count Number of units; must be at least 1.
     *  @throws std::invalid_argument if count is 0.
     */
    ResourcePool(std::string name, std::uint32_t count);

    /** Books the earliest-free unit.
     *
     *  @param earliest First cycle the operation may start.
     *  @param occupancy Cycles the unit is busy.
     *  @return The cycle the operation starts.
     */
    std::uint64_t reserve(std::uint64_t earliest, std::uint64_t occupancy);

    /** Name of the pool.
     *
     *  @return The name given to the constructor.
     */
    [[nodiscard]] const std::string& name() const { return name_; }

    /** Number of units.
     *
     *  @return The unit count.
     */
    [[nodiscard]] std::uint32_t count() const {
        return static_cast<std::uint32_t>(free_at_.size());
    }

    /** Operations booked so far.
     *
     *  @return Operation count.
     */
    [[nodiscard]] std::uint64_t ops() const { return ops_; }

    /** Busy cycles booked so far, summed over all units.
     *
     *  @return Busy cycles.
     */
    [[nodiscard]] std::uint64_t busy_cycles() const { return busy_cycles_; }

  private:
    /// Name used in statistics.
    std::string name_;
    /// Per unit: the first cycle it is free again.
    std::vector<std::uint64_t> free_at_;
    /// Operations booked.
    std::uint64_t ops_ = 0;
    /// Busy cycles booked, summed over all units.
    std::uint64_t busy_cycles_ = 0;
};

} // namespace reef_perf
