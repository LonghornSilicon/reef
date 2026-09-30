#include "Rob.hpp"

#include "sparta/utils/SpartaAssert.hpp"

namespace coralnpu_perf {

const char* Rob::name = "rob";

Rob::Rob(sparta::TreeNode* node, const RobParameterSet* p)
    : sparta::Unit(node, name),
      rob_entries_(p->rob_entries),
      retire_width_(p->retire_width) {
  sparta_assert(rob_entries_ > 0 && retire_width_ > 0);
  in_insts_.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Rob, receiveInst_, InstPtr));
  sparta::StartupEvent(node, CREATE_SPARTA_HANDLER(Rob, sendInitialCredits_));
}

void Rob::sendInitialCredits_() { out_credits_.send(rob_entries_); }

void Rob::receiveInst_(const InstPtr& inst) {
  rob_.push_back(inst);
  sparta_assert(rob_.size() <= rob_entries_, "Rob overflow");
  ev_retire_.schedule(0);
}

void Rob::retire_() {
  const uint64_t now = getClock()->currentCycle();
  uint32_t n = 0;
  while (n < retire_width_ && !rob_.empty() && rob_.front()->complete_cycle <= now) {
    rob_.front()->retire_cycle = now;
    last_retire_cycle_ = now;
    rob_.pop_front();
    ++n;
  }
  if (n > 0) {
    num_retired_ += n;
    out_credits_.send(n);
  }
  if (rob_.empty()) return;

  // Sleep until the oldest instruction can complete.
  const uint64_t head_done = rob_.front()->complete_cycle;
  if (head_done == Inst::kNever || head_done <= now) {
    ev_retire_.schedule(1);  // not timed by Execute yet, or retire bandwidth
  } else {
    ev_retire_.schedule(head_done - now);
  }
}

}  // namespace coralnpu_perf
