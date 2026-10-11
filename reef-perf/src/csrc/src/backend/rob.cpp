#include "reef_perf/backend/rob.hpp"

#include "sparta/events/StartupEvent.hpp"
#include "sparta/utils/SpartaAssert.hpp"

#include <cstdint>

namespace reef_perf {

Rob::Rob(sparta::TreeNode* node, const RobParameterSet* params)
    : sparta::Unit(node, name), rob_entries_(params->rob_entries),
      retire_width_(params->retire_width) {
    sparta_assert(rob_entries_ > 0 && retire_width_ > 0);
    in_insts_.registerConsumerHandler(
        CREATE_SPARTA_HANDLER_WITH_DATA(Rob, receive_inst, InstPtr));
    sparta::StartupEvent(node,
                         CREATE_SPARTA_HANDLER(Rob, send_initial_credits));
}

void Rob::send_initial_credits() { out_credits_.send(rob_entries_); }

void Rob::receive_inst(const InstPtr& inst) {
    rob_.push_back(inst);
    sparta_assert(rob_.size() <= rob_entries_, "Rob overflow");
    ev_retire_.schedule(); // this cycle
}

void Rob::retire_insts() {
    const std::uint64_t now = getClock()->currentCycle();
    std::uint32_t count = 0;
    while (count < retire_width_ && !rob_.empty() &&
           rob_.front()->complete_cycle <= now) {
        rob_.front()->retire_cycle = now;
        last_retire_cycle_ = now;
        rob_.pop_front();
        ++count;
    }
    if (count > 0) {
        num_retired_ += count;
        out_credits_.send(count);
    }
    if (rob_.empty()) {
        return;
    }

    // Sleep until the oldest instruction can complete.
    const std::uint64_t head_done = rob_.front()->complete_cycle;
    if (head_done == Inst::kNever || head_done <= now) {
        ev_retire_.schedule(1); // not timed by its unit yet, or retire width
    } else {
        ev_retire_.schedule(head_done - now);
    }
}

} // namespace reef_perf
