#include "reef_perf/matrix/mxu.hpp"

#include "sparta/events/StartupEvent.hpp"

#include <cstdint>

namespace reef_perf {

Mxu::Mxu(sparta::TreeNode* node, const MxuParameterSet* params)
    : sparta::Unit(node, name), queue_entries_(params->mtx_queue_entries),
      latency_(params->mtx_latency), occupancy_(params->mtx_occupancy),
      engines_("matrix", params->mtx_units) {
    in_insts_.registerConsumerHandler(
        CREATE_SPARTA_HANDLER_WITH_DATA(Mxu, receive_inst, InstPtr));
    sparta::StartupEvent(node,
                         CREATE_SPARTA_HANDLER(Mxu, send_initial_credits));
}

void Mxu::send_initial_credits() { out_credits_.send(queue_entries_); }

void Mxu::receive_inst(const InstPtr& inst) {
    const std::uint64_t now = getClock()->currentCycle();
    ++num_executed_;

    const std::uint64_t start = engines_.reserve(now, occupancy_);
    inst->issue_cycle = start;
    inst->result_ready_cycle = start + latency_ - 1;
    inst->complete_cycle = start + latency_;

    // The instruction leaves the command queue when it starts.
    out_credits_.send(1, start - now);
}

} // namespace reef_perf
