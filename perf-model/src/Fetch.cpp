#include "Fetch.hpp"

#include "sparta/utils/SpartaAssert.hpp"

namespace coralnpu_perf {

const char* Fetch::name = "fetch";

Fetch::Fetch(sparta::TreeNode* node, const FetchParameterSet* p)
    : sparta::Unit(node, name),
      fetch_width_(p->fetch_width),
      fetch_interval_(p->fetch_interval),
      redirect_penalty_(p->redirect_penalty) {
  sparta_assert(fetch_width_ > 0 && fetch_interval_ > 0);
  in_credits_.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Fetch, receiveCredits_, uint32_t));
  // Fetch starts when Dispatch sends its initial credits.
}

void Fetch::receiveCredits_(const uint32_t& credits) {
  credits_ += credits;
  scheduleFetch_();
}

void Fetch::scheduleFetch_() {
  if (done_ || credits_ == 0) return;
  const uint64_t now = getClock()->currentCycle();
  const uint64_t delay = next_fetch_cycle_ > now ? next_fetch_cycle_ - now : 0;
  ev_fetch_.schedule(delay);
}

void Fetch::fetch_() {
  sparta_assert(funcsim_ != nullptr, "FuncSim was not connected to Fetch");
  const uint64_t now = getClock()->currentCycle();
  if (done_) return;
  if (now < next_fetch_cycle_) {  // woken early by a credit; wait
    scheduleFetch_();
    return;
  }
  if (credits_ == 0) {
    ++cycles_no_credit_;
    return;  // receiveCredits_ will wake us up
  }

  FetchPacket pkt;
  bool redirect = false;
  while (pkt.insts.size() < fetch_width_ && credits_ > 0) {
    InstPtr inst = funcsim_->next();  // execute-at-fetch
    if (!inst) {
      done_ = true;
      pkt.last = true;
      break;
    }
    inst->fetch_cycle = now;
    pkt.insts.push_back(inst);
    --credits_;
    ++num_fetched_;
    if (inst->redirected()) {
      redirect = true;
      ++num_redirects_;
      break;  // the rest of this group would be on the wrong path
    }
  }

  if (!pkt.insts.empty() || pkt.last) {
    ++num_groups_;
    out_insts_.send(pkt);
  }
  next_fetch_cycle_ = now + fetch_interval_ + (redirect ? redirect_penalty_ : 0);
  scheduleFetch_();
}

}  // namespace coralnpu_perf
