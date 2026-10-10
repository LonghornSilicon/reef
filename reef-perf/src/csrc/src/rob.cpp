#include "reef_perf/rob.hpp"

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
    //Modified for ROI tracking
    while (count < retire_width_ && !rob_.empty() &&
        rob_.front()->complete_cycle <= now) {

        const auto& inst = rob_.front();
        const std::uint32_t encoding = inst->encoding;

        // Decode the CSR read instruction.
        const std::uint32_t opcode = encoding & 0x7F;
        const std::uint32_t funct3 = (encoding >> 12) & 0x7;
        const std::uint32_t csr = (encoding >> 20) & 0xFFF;
        const std::uint32_t rs1 = (encoding >> 15) & 0x1F;
        const std::uint32_t rd = (encoding >> 7) & 0x1F;

        const bool is_mcycle_read =
            opcode == 0x73 &&
            funct3 == 0x2 &&
            csr == 0xB00 &&
            rs1 == 0;

        if (is_mcycle_read && rd == 26) {
            // csrr s10, mcycle: begin ROI.
            roi_active_ = true;
            roi_start_cycle_ = now;
            roi_retired_insts_ = 0;
        } else if (is_mcycle_read && rd == 27) {
            // csrr s11, mcycle: end ROI.
            if (roi_active_) {
                roi_end_cycle_ = now;
                roi_active_ = false;
                roi_complete_ = true;
            }
        } else if (roi_active_) {
            // Count instructions strictly between markers.
            ++roi_retired_insts_;
        }

        inst->retire_cycle = now;
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
        ev_retire_.schedule(1); // not timed by Execute yet, or retire width
    } else {
        ev_retire_.schedule(head_done - now);
    }
}

} // namespace reef_perf
