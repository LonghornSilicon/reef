#include "reef_perf/mem/mem_timing.hpp"

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace reef_perf {

std::uint32_t distinct_blocks(std::span<const MemAccess> accesses,
                              std::uint32_t block_bytes) {
    if (block_bytes == 0) {
        throw std::invalid_argument("block size must be at least 1 byte");
    }
    if (accesses.empty()) {
        return 1;
    }
    std::unordered_set<std::uint32_t> blocks;
    for (const MemAccess& access : accesses) {
        // An access can straddle two blocks.
        blocks.insert(access.addr / block_bytes);
        blocks.insert((access.addr + access.size - 1) / block_bytes);
    }
    return static_cast<std::uint32_t>(blocks.size());
}

TcmModel::TcmModel(std::string name, std::uint32_t latency,
                   std::uint32_t line_bytes, std::uint32_t cycles_per_line)
    : latency_(latency), line_bytes_(line_bytes),
      cycles_per_line_(cycles_per_line), port_(std::move(name), 1) {}

MemResponse TcmModel::access(std::span<const MemAccess> accesses,
                             std::uint64_t earliest) {
    MemResponse resp;
    resp.occupancy =
        static_cast<std::uint64_t>(distinct_blocks(accesses, line_bytes_)) *
        cycles_per_line_;
    resp.latency = latency_ + resp.occupancy - 1;
    resp.start = port_.reserve(earliest, resp.occupancy);
    return resp;
}

AxiModel::AxiModel(std::uint32_t latency, std::uint32_t bytes_per_beat,
                   std::uint32_t max_outstanding)
    : latency_(latency), bytes_per_beat_(bytes_per_beat), data_("axi_data", 1),
      outstanding_("axi_outstanding", max_outstanding) {}

MemResponse AxiModel::access(std::span<const MemAccess> accesses,
                             std::uint64_t earliest) {
    MemResponse resp;
    resp.occupancy = distinct_blocks(accesses, bytes_per_beat_);
    resp.latency = latency_ + resp.occupancy - 1;
    // Start when both the data channel and an outstanding slot are free.
    resp.start =
        std::max(data_.next_free(earliest), outstanding_.next_free(earliest));
    data_.reserve(resp.start, resp.occupancy);
    outstanding_.reserve(resp.start, resp.latency);
    return resp;
}

} // namespace reef_perf
