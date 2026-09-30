// Execute-at-fetch driver: wraps MPACT's CoralNPUV2Simulator behind the C API
// in coralnpu_driver.h. See that header for the contract.

#include "perf_driver/coralnpu_driver.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "sim/coralnpu_v2_simulator.h"

namespace {

using ::coralnpu::sim::CoralNPUV2Simulator;
using ::coralnpu::sim::CoralNPUV2SimulatorOptions;
using ::mpact::sim::generic::DataBuffer;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::generic::ReferenceCount;
using ::mpact::sim::util::MemoryInterface;

// Sits between the processor state and memory, and records every access made
// while an instruction executes. Scalar loads/stores arrive as one access;
// vector loads/stores arrive as a list of per-element addresses plus a mask.
class RecordingMemory : public MemoryInterface {
 public:
  explicit RecordingMemory(MemoryInterface* inner) : inner_(inner) {}

  void Begin(cn_inst_t* rec) { rec_ = rec; }
  void End() { rec_ = nullptr; }

  void Load(uint64_t address, DataBuffer* db, Instruction* inst,
            ReferenceCount* context) override {
    Record(address, db->size<uint8_t>(), /*is_store=*/false);
    inner_->Load(address, db, inst, context);
  }
  void Load(DataBuffer* address_db, DataBuffer* mask_db, int el_size,
            DataBuffer* db, Instruction* inst,
            ReferenceCount* context) override {
    RecordVector(address_db, mask_db, el_size, /*is_store=*/false);
    inner_->Load(address_db, mask_db, el_size, db, inst, context);
  }
  void Store(uint64_t address, DataBuffer* db) override {
    Record(address, db->size<uint8_t>(), /*is_store=*/true);
    inner_->Store(address, db);
  }
  void Store(DataBuffer* address_db, DataBuffer* mask_db, int el_size,
             DataBuffer* db) override {
    RecordVector(address_db, mask_db, el_size, /*is_store=*/true);
    inner_->Store(address_db, mask_db, el_size, db);
  }

 private:
  void Record(uint64_t addr, int size, bool is_store) {
    if (rec_ == nullptr) return;
    if (rec_->num_mem >= CN_MAX_MEM) {
      rec_->mem_truncated = 1;
      return;
    }
    cn_mem_t& m = rec_->mem[rec_->num_mem++];
    m.addr = static_cast<uint32_t>(addr);
    m.size = static_cast<uint8_t>(size);
    m.is_store = is_store ? 1 : 0;
  }

  void RecordVector(DataBuffer* address_db, DataBuffer* mask_db, int el_size,
                    bool is_store) {
    if (rec_ == nullptr || address_db == nullptr) return;
    const int n = address_db->size<uint64_t>();
    const int n_mask = mask_db == nullptr ? 0 : mask_db->size<bool>();
    for (int i = 0; i < n; ++i) {
      const bool active = (i >= n_mask) || mask_db->Get<bool>(i);
      if (active) Record(address_db->Get<uint64_t>(i), el_size, is_store);
    }
  }

  MemoryInterface* inner_;
  cn_inst_t* rec_ = nullptr;
};

struct Driver {
  std::unique_ptr<CoralNPUV2Simulator> sim;
  std::unique_ptr<RecordingMemory> recmem;
  uint64_t seq = 0;
  bool halted = false;
};

}  // namespace

extern "C" {

void cn_default_options(cn_options_t* opts) {
  opts->itcm_start = 0x0;
  opts->itcm_length = 0x2000;   // 8 KB
  opts->dtcm_start = 0x10000;
  opts->dtcm_length = 0x8000;   // 32 KB
}

void* cn_create(const cn_options_t* opts, const char* elf_path) {
  cn_options_t o;
  if (opts != nullptr) {
    o = *opts;
  } else {
    cn_default_options(&o);
  }

  CoralNPUV2SimulatorOptions sim_opts;
  sim_opts.itcm_start_address = o.itcm_start;
  sim_opts.itcm_length = o.itcm_length;
  sim_opts.exit_on_ebreak = true;
  sim_opts.lsu_access_ranges.clear();
  sim_opts.lsu_access_ranges.push_back(
      {.start_address = o.dtcm_start, .length = o.dtcm_length});

  auto d = std::make_unique<Driver>();
  d->sim = std::make_unique<CoralNPUV2Simulator>(sim_opts);
  absl::Status st = d->sim->LoadProgram(elf_path);
  if (!st.ok()) {
    std::fprintf(stderr, "cn_create: failed to load '%s': %s\n", elf_path,
                 std::string(st.message()).c_str());
    return nullptr;
  }
  // Interpose the recorder between the state and whatever memory it uses.
  d->recmem = std::make_unique<RecordingMemory>(d->sim->state()->memory());
  d->sim->state()->set_memory(d->recmem.get());
  return d.release();
}

int cn_step(void* handle, cn_inst_t* out) {
  auto* d = static_cast<Driver*>(handle);
  if (d->halted) return 0;

  auto pc_or = d->sim->ReadRegister("pc");
  if (!pc_or.ok()) {
    std::fprintf(stderr, "cn_step: cannot read pc\n");
    return -1;
  }
  const uint32_t pc = static_cast<uint32_t>(*pc_or);

  out->seq = d->seq;
  out->pc = pc;
  out->encoding = 0;
  (void)d->sim->ReadMemory(pc, &out->encoding, sizeof(out->encoding));

  out->disasm[0] = '\0';
  auto dis_or = d->sim->top()->GetDisassembly(pc);
  if (dis_or.ok()) {
    std::snprintf(out->disasm, CN_DISASM_LEN, "%s", dis_or->c_str());
  }

  // Vector configuration in effect *before* this instruction executes.
  out->vl = 0;
  out->sew_bytes = 1;
  out->lmul8 = 8;
  if (auto* vs = d->sim->state()->rv_vector(); vs != nullptr) {
    out->vl = static_cast<uint32_t>(vs->vector_length());
    out->sew_bytes = static_cast<uint8_t>(vs->selected_element_width());
    out->lmul8 = static_cast<uint8_t>(vs->vector_length_multiplier());
  }

  out->num_mem = 0;
  out->mem_truncated = 0;
  d->recmem->Begin(out);
  auto steps = d->sim->Step(1);
  d->recmem->End();
  if (!steps.ok()) {
    std::fprintf(stderr, "cn_step: step failed at pc 0x%08x: %s\n", pc,
                 std::string(steps.status().message()).c_str());
    return -1;
  }
  if (*steps == 0) {
    d->halted = true;
    return 0;
  }

  auto next_or = d->sim->ReadRegister("pc");
  out->next_pc = next_or.ok() ? static_cast<uint32_t>(*next_or) : pc + 4;
  ++d->seq;

  // mpause (and ebreak, since exit_on_ebreak is set) end the program. MPACT
  // requests a halt while executing them; stop handing out instructions.
  if (std::strncmp(out->disasm, "mpause", 6) == 0 ||
      std::strncmp(out->disasm, "ebreak", 6) == 0) {
    d->halted = true;
  }
  return 1;
}

void cn_destroy(void* handle) { delete static_cast<Driver*>(handle); }

}  // extern "C"
