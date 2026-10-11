#include "reef_perf/backend/lsu.hpp"

#include "sparta/events/StartupEvent.hpp"
#include "sparta/utils/SpartaAssert.hpp"

#include <cstdint>

namespace reef_perf {

Lsu::Lsu(sparta::TreeNode* node, const LsuParameterSet* params)
    : sparta::Unit(node, name), queue_entries_(params->lsu_queue_entries),
      slot_("lsu", 1) {
    in_insts_.registerConsumerHandler(
        CREATE_SPARTA_HANDLER_WITH_DATA(Lsu, receive_inst, InstPtr));
    sparta::StartupEvent(node,
                         CREATE_SPARTA_HANDLER(Lsu, send_initial_credits));
}

void Lsu::send_initial_credits() { out_credits_.send(queue_entries_); }

void Lsu::receive_inst(const InstPtr& inst) {
    sparta_assert(memory_ != nullptr, "memory was not connected to the LSU");
    const std::uint64_t now = getClock()->currentCycle();
    ++num_executed_;

    MemRequest request;
    request.requester = Requester::LSU;
    request.accesses = inst->mem;
    request.earliest = slot_.next_free(now);
    const MemResponse resp = memory_->access(request);

    const std::uint64_t start = slot_.reserve(resp.start, resp.occupancy);
    sparta_assert(start == resp.start, "LSU slot busy at the memory's start");
    inst->issue_cycle = start;
    inst->result_ready_cycle = start + resp.latency - 1;
    inst->complete_cycle = start + resp.latency;

    // The instruction leaves the queue when it starts.
    out_credits_.send(1, start - now);
}

} // namespace reef_perf
