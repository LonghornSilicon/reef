#include "reef_perf/backend/scalar_exec.hpp"

#include <cstdint>
#include <vector>

namespace reef_perf {

ScalarExecConfig ScalarExec::read_config(const ScalarExecParameterSet* params) {
    ScalarExecConfig cfg;
    cfg.alu_count = params->alu_count;
    cfg.alu_latency = params->alu_latency;
    cfg.mul_count = params->mul_count;
    cfg.mul_latency = params->mul_latency;
    cfg.mul_occupancy = params->mul_occupancy;
    cfg.div_latency = params->div_latency;
    cfg.div_occupancy = params->div_occupancy;
    cfg.fpu_latency = params->fpu_latency;
    cfg.fpu_occupancy = params->fpu_occupancy;
    cfg.fdiv_latency = params->fdiv_latency;
    cfg.fdiv_occupancy = params->fdiv_occupancy;
    return cfg;
}

ScalarExec::ScalarExec(sparta::TreeNode* node,
                       const ScalarExecParameterSet* params)
    : sparta::Unit(node, name), cfg_(read_config(params)),
      alu_("alu", cfg_.alu_count), mul_("mul", cfg_.mul_count), div_("div", 1),
      fpu_("fpu", 1), fdiv_("fdiv", 1) {
    in_insts_.registerConsumerHandler(
        CREATE_SPARTA_HANDLER_WITH_DATA(ScalarExec, receive_inst, InstPtr));
}

void ScalarExec::receive_inst(const InstPtr& inst) {
    const std::uint64_t now = getClock()->currentCycle();
    ++num_executed_;

    ResourcePool* pool = &alu_;
    std::uint64_t occupancy = 1;
    std::uint64_t latency = cfg_.alu_latency;
    switch (inst->cls) {
    case InstClass::MUL:
        pool = &mul_;
        occupancy = cfg_.mul_occupancy;
        latency = cfg_.mul_latency;
        break;
    case InstClass::DIV:
        pool = &div_;
        occupancy = cfg_.div_occupancy;
        latency = cfg_.div_latency;
        break;
    case InstClass::FP:
        pool = &fpu_;
        occupancy = cfg_.fpu_occupancy;
        latency = cfg_.fpu_latency;
        break;
    case InstClass::FP_DIV:
        pool = &fdiv_;
        occupancy = cfg_.fdiv_occupancy;
        latency = cfg_.fdiv_latency;
        break;
    case InstClass::UNKNOWN:
        ++num_unknown_;
        break;
    default: // ALU, BRANCH, JUMP, CSR, FENCE, SYSTEM
        break;
    }

    const std::uint64_t start = pool->reserve(now, occupancy);
    inst->issue_cycle = start;
    inst->result_ready_cycle = start + latency - 1;
    inst->complete_cycle = start + latency;
}

std::vector<const ResourcePool*> ScalarExec::pools() const {
    return {&alu_, &mul_, &div_, &fpu_, &fdiv_};
}

} // namespace reef_perf
