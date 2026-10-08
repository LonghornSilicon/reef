#include "reef_perf/backend/dispatch.hpp"

#include "sparta/events/StartupEvent.hpp"
#include "sparta/utils/SpartaAssert.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace reef_perf {

const char* stall_reason_name(StallReason reason) {
    switch (reason) {
    case StallReason::IBUF_EMPTY:
        return "ibuf_empty";
    case StallReason::RAW:
        return "raw_hazard";
    case StallReason::WAW:
        return "waw_hazard";
    case StallReason::ROB_FULL:
        return "rob_full";
    case StallReason::LSU_FULL:
        return "lsu_queue_full";
    case StallReason::VEC_FULL:
        return "vector_queue_full";
    case StallReason::NUM_REASONS:
        break;
    }
    return "unknown";
}

Dispatch::Dispatch(sparta::TreeNode* node, const DispatchParameterSet* params)
    : sparta::Unit(node, name), dispatch_width_(params->dispatch_width),
      ibuf_entries_(params->ibuf_entries) {
    sparta_assert(dispatch_width_ > 0 && ibuf_entries_ > 0);

    constexpr auto kReasons =
        static_cast<std::size_t>(StallReason::NUM_REASONS);
    for (std::size_t i = 0; i < kReasons; ++i) {
        const auto reason = static_cast<StallReason>(i);
        stall_counters_.push_back(std::make_unique<sparta::Counter>(
            &unit_stat_set_, std::string("stall_") + stall_reason_name(reason),
            std::string("Cycles with no dispatch, blamed on: ") +
                stall_reason_name(reason),
            sparta::Counter::COUNT_NORMAL));
    }

    in_insts_.registerConsumerHandler(
        CREATE_SPARTA_HANDLER_WITH_DATA(Dispatch, receive_insts, FetchPacket));
    in_rob_credits_.registerConsumerHandler(CREATE_SPARTA_HANDLER_WITH_DATA(
        Dispatch, receive_rob_credits, std::uint32_t));
    in_lsu_credits_.registerConsumerHandler(CREATE_SPARTA_HANDLER_WITH_DATA(
        Dispatch, receive_lsu_credits, std::uint32_t));
    in_vec_credits_.registerConsumerHandler(CREATE_SPARTA_HANDLER_WITH_DATA(
        Dispatch, receive_vec_credits, std::uint32_t));
    sparta::StartupEvent(node,
                         CREATE_SPARTA_HANDLER(Dispatch, send_initial_credits));
}

void Dispatch::send_initial_credits() {
    out_fetch_credits_.send(ibuf_entries_);
}

void Dispatch::receive_insts(const FetchPacket& pkt) {
    for (const auto& inst : pkt.insts) {
        ibuf_.push_back(inst);
    }
    sparta_assert(ibuf_.size() <= ibuf_entries_, "instruction buffer overflow");
    if (pkt.last) {
        fetch_done_ = true;
    }
    schedule_dispatch();
}

void Dispatch::receive_rob_credits(const std::uint32_t& credits) {
    rob_credits_ += credits;
    schedule_dispatch();
}

void Dispatch::receive_lsu_credits(const std::uint32_t& credits) {
    lsu_credits_ += credits;
    schedule_dispatch();
}

void Dispatch::receive_vec_credits(const std::uint32_t& credits) {
    vec_credits_ += credits;
    schedule_dispatch();
}

void Dispatch::schedule_dispatch() {
    // Keep ticking every cycle until the program is fully dispatched, so
    // cycles with an empty buffer are counted as IBUF_EMPTY stalls.
    if (!ibuf_.empty() || !fetch_done_) {
        ev_dispatch_.schedule(); // this cycle
    }
}

bool Dispatch::can_dispatch(const InstPtr& inst, std::uint64_t now,
                            StallReason& why) const {
    for (const std::uint16_t reg : inst->srcs) {
        const InstPtr& writer = last_writer_.at(reg);
        if (writer && writer->result_ready_cycle > now) {
            why = StallReason::RAW;
            return false;
        }
    }
    for (const std::uint16_t reg : inst->dsts) {
        const InstPtr& writer = last_writer_.at(reg);
        if (writer && writer->result_ready_cycle > now) {
            why = StallReason::WAW;
            return false;
        }
    }
    if (rob_credits_ == 0) {
        why = StallReason::ROB_FULL;
        return false;
    }
    if (inst->is_memory() && lsu_credits_ == 0) {
        why = StallReason::LSU_FULL;
        return false;
    }
    if (inst->is_vector() && !inst->is_memory() && vec_credits_ == 0) {
        why = StallReason::VEC_FULL;
        return false;
    }
    return true;
}

void Dispatch::dispatch_group() {
    const std::uint64_t now = getClock()->currentCycle();
    std::uint32_t count = 0;
    StallReason why = StallReason::IBUF_EMPTY;

    while (count < dispatch_width_ && !ibuf_.empty()) {
        const InstPtr inst = ibuf_.front();
        if (!can_dispatch(inst, now, why)) {
            break;
        }
        inst->dispatch_cycle = now;
        for (const std::uint16_t reg : inst->dsts) {
            last_writer_.at(reg) = inst;
        }
        --rob_credits_;
        if (inst->is_memory()) {
            --lsu_credits_;
        } else if (inst->is_vector()) {
            --vec_credits_;
        }
        out_execute_.send(inst);
        out_rob_.send(inst);
        ibuf_.pop_front();
        ++count;
    }

    if (count > 0) {
        num_dispatched_ += count;
        ++cycles_with_dispatch_;
        out_fetch_credits_.send(count);
    } else {
        ++(*stall_counters_.at(static_cast<std::size_t>(why)));
    }

    if (!ibuf_.empty() || !fetch_done_) {
        ev_dispatch_.schedule(1);
    }
}

} // namespace reef_perf
