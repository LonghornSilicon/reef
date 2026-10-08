#include "reef_perf/backend/execute.hpp"

#include "sparta/events/StartupEvent.hpp"

#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace reef_perf {

ExecuteConfig Execute::read_config(const ExecuteParameterSet* params) {
    ExecuteConfig cfg;
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
    cfg.lsu_queue_entries = params->lsu_queue_entries;
    cfg.lsu_latency = params->lsu_latency;
    cfg.lsu_line_bytes = params->lsu_line_bytes;
    cfg.lsu_cycles_per_line = params->lsu_cycles_per_line;
    cfg.vec_queue_entries = params->vec_queue_entries;
    cfg.vec_units = params->vec_units;
    cfg.vlen_bits = params->vlen_bits;
    cfg.vec_latency = params->vec_latency;
    cfg.vec_cycles_per_uop = params->vec_cycles_per_uop;
    cfg.vec_div_cycles_per_uop = params->vec_div_cycles_per_uop;
    cfg.vec_to_scalar_latency = params->vec_to_scalar_latency;
    return cfg;
}

Execute::Execute(sparta::TreeNode* node, const ExecuteParameterSet* params)
    : sparta::Unit(node, name), cfg_(read_config(params)),
      alu_("alu", cfg_.alu_count), mul_("mul", cfg_.mul_count), div_("div", 1),
      fpu_("fpu", 1), fdiv_("fdiv", 1), lsu_("lsu", 1),
      vec_("vector", cfg_.vec_units) {
    in_insts_.registerConsumerHandler(
        CREATE_SPARTA_HANDLER_WITH_DATA(Execute, receive_inst, InstPtr));
    sparta::StartupEvent(node,
                         CREATE_SPARTA_HANDLER(Execute, send_initial_credits));
}

void Execute::send_initial_credits() {
    out_lsu_credits_.send(cfg_.lsu_queue_entries);
    out_vec_credits_.send(cfg_.vec_queue_entries);
}

std::uint32_t Execute::distinct_lines(const Inst& inst) const {
    if (inst.mem.empty()) {
        return 1;
    }
    const std::uint32_t line_bytes = cfg_.lsu_line_bytes;
    std::unordered_set<std::uint32_t> lines;
    for (const MemAccess& access : inst.mem) {
        // An access can straddle two lines.
        lines.insert(access.addr / line_bytes);
        lines.insert((access.addr + access.size - 1) / line_bytes);
    }
    return static_cast<std::uint32_t>(lines.size());
}

std::uint32_t Execute::vector_uops(const Inst& inst) const {
    const std::uint32_t vlenb = cfg_.vlen_bits / 8;
    const std::uint32_t bytes = inst.vl * inst.sew_bytes;
    return std::max<std::uint32_t>(1, (bytes + vlenb - 1) / vlenb);
}

void Execute::receive_inst(const InstPtr& inst) {
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
    case InstClass::LOAD:
    case InstClass::STORE:
    case InstClass::FP_LOAD:
    case InstClass::FP_STORE:
    case InstClass::V_LOAD:
    case InstClass::V_STORE:
        pool = &lsu_;
        occupancy = static_cast<std::uint64_t>(distinct_lines(*inst)) *
                    cfg_.lsu_cycles_per_line;
        latency = cfg_.lsu_latency + occupancy - 1;
        break;
    case InstClass::V_ALU:
    case InstClass::V_MUL:
    case InstClass::V_FP:
    case InstClass::V_PERM:
    case InstClass::V_TO_SCALAR:
        pool = &vec_;
        occupancy = static_cast<std::uint64_t>(vector_uops(*inst)) *
                    cfg_.vec_cycles_per_uop;
        latency = cfg_.vec_latency + occupancy - 1;
        if (inst->cls == InstClass::V_TO_SCALAR) {
            latency += cfg_.vec_to_scalar_latency;
        }
        break;
    case InstClass::V_DIV:
    case InstClass::V_FDIV:
        pool = &vec_;
        occupancy = static_cast<std::uint64_t>(vector_uops(*inst)) *
                    cfg_.vec_div_cycles_per_uop;
        latency = cfg_.vec_latency + occupancy - 1;
        break;
    case InstClass::UNKNOWN:
        ++num_unknown_;
        break;
    default: // ALU, BRANCH, JUMP, CSR, FENCE, SYSTEM, VSET
        break;
    }

    const std::uint64_t start = pool->reserve(now, occupancy);
    inst->issue_cycle = start;
    inst->result_ready_cycle = start + latency - 1;
    inst->complete_cycle = start + latency;

    // The instruction leaves its queue when it starts; return the credit
    // then. These conditions must match the ones Dispatch uses.
    if (inst->is_memory()) {
        out_lsu_credits_.send(1, start - now);
    } else if (inst->is_vector()) {
        out_vec_credits_.send(1, start - now);
    }
}

std::vector<const ResourcePool*> Execute::pools() const {
    return {&alu_, &mul_, &div_, &fpu_, &fdiv_, &lsu_, &vec_};
}

} // namespace reef_perf
