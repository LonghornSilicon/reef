#include "reef_perf/fetch.hpp"

#include "sparta/utils/SpartaAssert.hpp"

#include <cstdint>

namespace reef_perf {

Fetch::Fetch(sparta::TreeNode* node, const FetchParameterSet* params)
    : sparta::Unit(node, name), fetch_width_(params->fetch_width),
      fetch_interval_(params->fetch_interval),
      redirect_penalty_(params->redirect_penalty) {
    sparta_assert(fetch_width_ > 0 && fetch_interval_ > 0);
    in_credits_.registerConsumerHandler(
        CREATE_SPARTA_HANDLER_WITH_DATA(Fetch, receive_credits, std::uint32_t));
    // Fetch starts when Dispatch sends its initial credits.
}

void Fetch::receive_credits(const std::uint32_t& credits) {
    credits_ += credits;
    schedule_fetch();
}

void Fetch::schedule_fetch() {
    if (done_ || credits_ == 0) {
        return;
    }
    const std::uint64_t now = getClock()->currentCycle();
    const std::uint64_t delay =
        next_fetch_cycle_ > now ? next_fetch_cycle_ - now : 0;
    ev_fetch_.schedule(delay);
}

void Fetch::fetch_group() {
    sparta_assert(funcsim_ != nullptr, "FuncSim was not connected to Fetch");
    const std::uint64_t now = getClock()->currentCycle();
    if (done_) {
        return;
    }
    if (now < next_fetch_cycle_) { // woken early by a credit; wait
        schedule_fetch();
        return;
    }
    if (credits_ == 0) {
        ++cycles_no_credit_;
        return; // receive_credits wakes us up
    }

    FetchPacket pkt;
    bool redirect = false;
    while (pkt.insts.size() < fetch_width_ && credits_ > 0) {
        InstPtr inst = funcsim_->next(); // execute-at-fetch
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
            break; // the rest of this group would be on the wrong path
        }
    }

    if (!pkt.insts.empty() || pkt.last) {
        ++num_groups_;
        out_insts_.send(pkt);
    }
    next_fetch_cycle_ =
        now + fetch_interval_ + (redirect ? redirect_penalty_ : 0);
    schedule_fetch();
}

} // namespace reef_perf
