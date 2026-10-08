#include "reef_perf/common/resource_pool.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace reef_perf {

ResourcePool::ResourcePool(std::string name, std::uint32_t count)
    : name_(std::move(name)), free_at_(count, 0) {
    if (count == 0) {
        throw std::invalid_argument("pool '" + name_ +
                                    "' needs at least one unit");
    }
}

std::uint64_t ResourcePool::reserve(std::uint64_t earliest,
                                    std::uint64_t occupancy) {
    auto unit = std::ranges::min_element(free_at_);
    const std::uint64_t start = std::max(earliest, *unit);
    *unit = start + occupancy;
    ++ops_;
    busy_cycles_ += occupancy;
    return start;
}

} // namespace reef_perf
