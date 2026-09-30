#include "Execute.hpp"

#include <algorithm>
#include <unordered_set>

#include "sparta/utils/SpartaAssert.hpp"

namespace coralnpu_perf {

const char* Execute::name = "execute";

uint64_t Execute::Pool::reserve(uint64_t earliest, uint64_t occupancy) {
  auto it = std::min_element(free_at.begin(), free_at.end());
  const uint64_t start = std::max(earliest, *it);
  *it = start + occupancy;
  ++ops;
  busy_cycles += occupancy;
  return start;
}

namespace {
Execute::Pool makePool(const std::string& name, uint32_t count) {
  sparta_assert(count > 0, "pool '" << name << "' needs at least one unit");
  Execute::Pool pool;
  pool.name = name;
  pool.free_at.assign(count, 0);
  return pool;
}
}  // namespace

Execute::Execute(sparta::TreeNode* node, const ExecuteParameterSet* p)
    : sparta::Unit(node, name),
      p_(p),
      alu_(makePool("alu", p->alu_count)),
      mul_(makePool("mul", p->mul_count)),
      div_(makePool("div", 1)),
      fpu_(makePool("fpu", 1)),
      fdiv_(makePool("fdiv", 1)),
      lsu_(makePool("lsu", 1)),
      vec_(makePool("vector", p->vec_units)) {
  in_insts_.registerConsumerHandler(
      CREATE_SPARTA_HANDLER_WITH_DATA(Execute, receiveInst_, InstPtr));
  sparta::StartupEvent(node, CREATE_SPARTA_HANDLER(Execute, sendInitialCredits_));
}

void Execute::sendInitialCredits_() {
  out_lsu_credits_.send(p_->lsu_queue_entries);
  out_vec_credits_.send(p_->vec_queue_entries);
}

uint32_t Execute::distinctLines_(const Inst& inst) const {
  if (inst.mem.empty()) return 1;
  std::unordered_set<uint32_t> lines;
  for (const auto& m : inst.mem) {
    // An access can straddle two lines.
    lines.insert(m.addr / p_->lsu_line_bytes);
    lines.insert((m.addr + m.size - 1) / p_->lsu_line_bytes);
  }
  return static_cast<uint32_t>(lines.size());
}

uint32_t Execute::vectorUops_(const Inst& inst) const {
  const uint32_t vlenb = p_->vlen_bits / 8;
  const uint32_t bytes = inst.vl * inst.sew_bytes;
  return std::max<uint32_t>(1, (bytes + vlenb - 1) / vlenb);
}

void Execute::receiveInst_(const InstPtr& inst) {
  const uint64_t now = getClock()->currentCycle();
  ++num_executed_;

  Pool* pool = &alu_;
  uint64_t occupancy = 1;
  uint64_t latency = p_->alu_latency;

  switch (inst->cls) {
    case InstClass::MUL:
      pool = &mul_;
      occupancy = p_->mul_occupancy;
      latency = p_->mul_latency;
      break;
    case InstClass::DIV:
      pool = &div_;
      occupancy = p_->div_occupancy;
      latency = p_->div_latency;
      break;
    case InstClass::FP:
      pool = &fpu_;
      occupancy = p_->fpu_occupancy;
      latency = p_->fpu_latency;
      break;
    case InstClass::FP_DIV:
      pool = &fdiv_;
      occupancy = p_->fdiv_occupancy;
      latency = p_->fdiv_latency;
      break;
    case InstClass::LOAD:
    case InstClass::STORE:
    case InstClass::FP_LOAD:
    case InstClass::FP_STORE:
    case InstClass::V_LOAD:
    case InstClass::V_STORE:
      pool = &lsu_;
      occupancy = static_cast<uint64_t>(distinctLines_(*inst)) * p_->lsu_cycles_per_line;
      latency = p_->lsu_latency + occupancy - 1;
      break;
    case InstClass::V_ALU:
    case InstClass::V_MUL:
    case InstClass::V_FP:
    case InstClass::V_PERM:
    case InstClass::V_TO_SCALAR: {
      pool = &vec_;
      const uint64_t uops = vectorUops_(*inst);
      occupancy = uops * p_->vec_cycles_per_uop;
      latency = p_->vec_latency + occupancy - 1;
      if (inst->cls == InstClass::V_TO_SCALAR) latency += p_->vec_to_scalar_latency;
      break;
    }
    case InstClass::V_DIV:
    case InstClass::V_FDIV: {
      pool = &vec_;
      const uint64_t uops = vectorUops_(*inst);
      occupancy = uops * p_->vec_div_cycles_per_uop;
      latency = p_->vec_latency + occupancy - 1;
      break;
    }
    case InstClass::UNKNOWN:
      ++num_unknown_;
      break;
    default:  // ALU, BRANCH, JUMP, CSR, FENCE, SYSTEM, VSET
      break;
  }

  const uint64_t start = pool->reserve(now, occupancy);
  inst->issue_cycle = start;
  inst->result_ready_cycle = start + latency - 1;
  inst->complete_cycle = start + latency;

  // The instruction leaves its queue when it starts; return the credit then.
  // These conditions must match the ones Dispatch uses to take credits.
  if (inst->isMemory()) {
    out_lsu_credits_.send(1, start - now);
  } else if (inst->isVector()) {
    out_vec_credits_.send(1, start - now);
  }
}

std::vector<Execute::PoolStats> Execute::poolStats() const {
  std::vector<PoolStats> out;
  for (const Pool* pool : {&alu_, &mul_, &div_, &fpu_, &fdiv_, &lsu_, &vec_}) {
    out.push_back({pool->name, static_cast<uint32_t>(pool->free_at.size()),
                   pool->ops, pool->busy_cycles});
  }
  return out;
}

}  // namespace coralnpu_perf
