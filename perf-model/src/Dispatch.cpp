#include "Dispatch.hpp"

#include <string>

#include "sparta/events/StartupEvent.hpp"
#include "sparta/utils/SpartaAssert.hpp"

namespace coralnpu_perf {

const char* Dispatch::name = "dispatch";

const char* stallReasonName(StallReason r) {
  switch (r) {
    case StallReason::IBUF_EMPTY: return "ibuf_empty";
    case StallReason::RAW: return "raw_hazard";
    case StallReason::WAW: return "waw_hazard";
    case StallReason::ROB_FULL: return "rob_full";
    case StallReason::LSU_FULL: return "lsu_queue_full";
    case StallReason::VEC_FULL: return "vector_queue_full";
    default: return "unknown";
  }
}

Dispatch::Dispatch(sparta::TreeNode* node, const DispatchParameterSet* p)
    : sparta::Unit(node, name),
      dispatch_width_(p->dispatch_width),
      ibuf_entries_(p->ibuf_entries) {
  sparta_assert(dispatch_width_ > 0 && ibuf_entries_ > 0);

  for (size_t i = 0; i < static_cast<size_t>(StallReason::NUM_REASONS); ++i) {
    const auto r = static_cast<StallReason>(i);
    stall_counters_.emplace_back(new sparta::Counter(
        &unit_stat_set_, std::string("stall_") + stallReasonName(r),
        std::string("Cycles with no dispatch, blamed on: ") + stallReasonName(r),
        sparta::Counter::COUNT_NORMAL));
  }

  in_insts_.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Dispatch, receiveInsts_, FetchPacket));
  in_rob_credits_.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Dispatch, receiveRobCredits_, uint32_t));
  in_lsu_credits_.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Dispatch, receiveLsuCredits_, uint32_t));
  in_vec_credits_.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Dispatch, receiveVecCredits_, uint32_t));
  sparta::StartupEvent(node, CREATE_SPARTA_HANDLER(Dispatch, sendInitialCredits_));
}

void Dispatch::sendInitialCredits_() { out_fetch_credits_.send(ibuf_entries_); }

void Dispatch::receiveInsts_(const FetchPacket& pkt) {
  for (const auto& inst : pkt.insts) ibuf_.push_back(inst);
  sparta_assert(ibuf_.size() <= ibuf_entries_, "instruction buffer overflow");
  if (pkt.last) fetch_done_ = true;
  scheduleDispatch_();
}

void Dispatch::receiveRobCredits_(const uint32_t& n) { rob_credits_ += n; scheduleDispatch_(); }
void Dispatch::receiveLsuCredits_(const uint32_t& n) { lsu_credits_ += n; scheduleDispatch_(); }
void Dispatch::receiveVecCredits_(const uint32_t& n) { vec_credits_ += n; scheduleDispatch_(); }

void Dispatch::scheduleDispatch_() {
  // Keep ticking every cycle until the program is fully dispatched, so that
  // cycles with an empty buffer are counted as IBUF_EMPTY stalls.
  if (!ibuf_.empty() || !fetch_done_) ev_dispatch_.schedule(0);
}

bool Dispatch::canDispatch_(const InstPtr& inst, uint64_t now, StallReason& why) const {
  for (uint16_t r : inst->srcs) {
    const InstPtr& w = last_writer_[r];
    if (w && w->result_ready_cycle > now) { why = StallReason::RAW; return false; }
  }
  for (uint16_t r : inst->dsts) {
    const InstPtr& w = last_writer_[r];
    if (w && w->result_ready_cycle > now) { why = StallReason::WAW; return false; }
  }
  if (rob_credits_ == 0) { why = StallReason::ROB_FULL; return false; }
  if (inst->isMemory() && lsu_credits_ == 0) { why = StallReason::LSU_FULL; return false; }
  if (inst->isVector() && !inst->isMemory() && vec_credits_ == 0) {
    why = StallReason::VEC_FULL;
    return false;
  }
  return true;
}

void Dispatch::dispatch_() {
  const uint64_t now = getClock()->currentCycle();
  uint32_t n = 0;
  StallReason why = StallReason::IBUF_EMPTY;

  while (n < dispatch_width_ && !ibuf_.empty()) {
    const InstPtr inst = ibuf_.front();
    if (!canDispatch_(inst, now, why)) break;

    inst->dispatch_cycle = now;
    for (uint16_t r : inst->dsts) last_writer_[r] = inst;
    --rob_credits_;
    if (inst->isMemory()) {
      --lsu_credits_;
    } else if (inst->isVector()) {
      --vec_credits_;
    }
    out_execute_.send(inst);
    out_rob_.send(inst);
    ibuf_.pop_front();
    ++n;
  }

  if (n > 0) {
    num_dispatched_ += n;
    ++cycles_with_dispatch_;
    out_fetch_credits_.send(n);
  } else {
    ++(*stall_counters_[static_cast<size_t>(why)]);
  }

  if (!ibuf_.empty() || !fetch_done_) ev_dispatch_.schedule(1);
}

}  // namespace coralnpu_perf
