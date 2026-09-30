#include "FuncSim.hpp"

#include <stdexcept>

#include "InstDecode.hpp"

namespace coralnpu_perf {

FuncSim::FuncSim(const std::string& elf_path, const cn_options_t& options)
    : rec_(std::make_unique<cn_inst_t>()) {
  handle_ = cn_create(&options, elf_path.c_str());
  if (handle_ == nullptr) {
    throw std::runtime_error("MPACT could not load ELF: " + elf_path);
  }
}

FuncSim::~FuncSim() {
  if (handle_ != nullptr) cn_destroy(handle_);
}

InstPtr FuncSim::next() {
  if (halted_) return nullptr;
  const int rc = cn_step(handle_, rec_.get());
  if (rc < 0) {
    throw std::runtime_error("MPACT reported an error while stepping");
  }
  if (rc == 0) {
    halted_ = true;
    return nullptr;
  }

  const cn_inst_t& r = *rec_;
  auto inst = std::make_shared<Inst>();
  inst->seq = r.seq;
  inst->pc = r.pc;
  inst->next_pc = r.next_pc;
  inst->encoding = r.encoding;
  inst->disasm = r.disasm;
  inst->vl = r.vl;
  inst->sew_bytes = r.sew_bytes;
  inst->lmul8 = r.lmul8;
  inst->mem.reserve(r.num_mem);
  for (uint16_t i = 0; i < r.num_mem; ++i) {
    inst->mem.push_back({r.mem[i].addr, r.mem[i].size, r.mem[i].is_store != 0});
  }
  decodeInst(*inst);
  ++executed_;
  return inst;
}

}  // namespace coralnpu_perf
