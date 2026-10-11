#include "reef_perf/vector/vxu.hpp"

#include "sparta/events/StartupEvent.hpp"
#include "sparta/utils/SpartaAssert.hpp"

#include <algorithm>
#include <cstdint>

namespace reef_perf {

std::uint32_t vector_uops(const Inst& inst, std::uint32_t vlen_bits) {
    const std::uint32_t vlenb = vlen_bits / 8;
    const std::uint32_t bytes = inst.vl * inst.sew_bytes;
    return std::max<std::uint32_t>(1, (bytes + vlenb - 1) / vlenb);
}

VxuConfig Vxu::read_config(const VxuParameterSet* params) {
    VxuConfig cfg;
    cfg.vec_queue_entries = params->vec_queue_entries;
    cfg.vec_units = params->vec_units;
    cfg.vlen_bits = params->vlen_bits;
    cfg.vec_latency = params->vec_latency;
    cfg.vec_cycles_per_uop = params->vec_cycles_per_uop;
    cfg.vec_div_cycles_per_uop = params->vec_div_cycles_per_uop;
    cfg.vec_to_scalar_latency = params->vec_to_scalar_latency;
    cfg.vset_latency = params->vset_latency;
    return cfg;
}

Vxu::Vxu(sparta::TreeNode* node, const VxuParameterSet* params)
    : sparta::Unit(node, name), cfg_(read_config(params)),
      lanes_("vector", cfg_.vec_units) {
    sparta_assert(cfg_.vlen_bits >= 8);
    in_insts_.registerConsumerHandler(
        CREATE_SPARTA_HANDLER_WITH_DATA(Vxu, receive_inst, InstPtr));
    sparta::StartupEvent(node,
                         CREATE_SPARTA_HANDLER(Vxu, send_initial_credits));
}

void Vxu::send_initial_credits() { out_credits_.send(cfg_.vec_queue_entries); }

void Vxu::receive_inst(const InstPtr& inst) {
    const std::uint64_t now = getClock()->currentCycle();
    ++num_executed_;

    std::uint64_t start = now;
    std::uint64_t latency = cfg_.vset_latency;
    if (inst->cls != InstClass::VSET) {
        const bool divide =
            inst->cls == InstClass::V_DIV || inst->cls == InstClass::V_FDIV;
        const std::uint64_t per_uop =
            divide ? cfg_.vec_div_cycles_per_uop : cfg_.vec_cycles_per_uop;
        const std::uint64_t occupancy =
            static_cast<std::uint64_t>(vector_uops(*inst, cfg_.vlen_bits)) *
            per_uop;
        latency = cfg_.vec_latency + occupancy - 1;
        if (inst->cls == InstClass::V_TO_SCALAR) {
            latency += cfg_.vec_to_scalar_latency;
        }
        start = lanes_.reserve(now, occupancy);
    }
    inst->issue_cycle = start;
    inst->result_ready_cycle = start + latency - 1;
    inst->complete_cycle = start + latency;

    // The instruction leaves the command queue when it starts.
    out_credits_.send(1, start - now);
}

} // namespace reef_perf
