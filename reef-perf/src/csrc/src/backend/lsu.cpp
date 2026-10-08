#include "reef_perf/backend/lsu.hpp"

#include "sparta/events/StartupEvent.hpp"
#include "sparta/utils/SpartaAssert.hpp"

#include <cstdint>
#include <unordered_set>

namespace reef_perf {

Lsu::Lsu(sparta::TreeNode* node, const LsuParameterSet* params)
    : sparta::Unit(node, name), queue_entries_(params->lsu_queue_entries),
      latency_(params->lsu_latency), line_bytes_(params->lsu_line_bytes),
      cycles_per_line_(params->lsu_cycles_per_line), slot_("lsu", 1) {
    sparta_assert(line_bytes_ > 0);
    in_insts_.registerConsumerHandler(
        CREATE_SPARTA_HANDLER_WITH_DATA(Lsu, receive_inst, InstPtr));
    sparta::StartupEvent(node, CREATE_SPARTA_HANDLER(Lsu, send_initial_credits));
}

void Lsu::send_initial_credits() { out_credits_.send(queue_entries_); }

std::uint32_t Lsu::distinct_lines(const Inst& inst) const {
    if (inst.mem.empty()) {
        return 1;
    }
    std::unordered_set<std::uint32_t> lines;
    for (const MemAccess& access : inst.mem) {
        // An access can straddle two lines.
        lines.insert(access.addr / line_bytes_);
        lines.insert((access.addr + access.size - 1) / line_bytes_);
    }
    return static_cast<std::uint32_t>(lines.size());
}

void Lsu::receive_inst(const InstPtr& inst) {
    const std::uint64_t now = getClock()->currentCycle();
    ++num_executed_;

    const std::uint64_t occupancy =
        static_cast<std::uint64_t>(distinct_lines(*inst)) * cycles_per_line_;
    const std::uint64_t latency = latency_ + occupancy - 1;
    const std::uint64_t start = slot_.reserve(now, occupancy);
    inst->issue_cycle = start;
    inst->result_ready_cycle = start + latency - 1;
    inst->complete_cycle = start + latency;

    // The instruction leaves the queue when it starts.
    out_credits_.send(1, start - now);
}

} // namespace reef_perf
