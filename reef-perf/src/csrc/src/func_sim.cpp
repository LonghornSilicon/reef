#include "reef_perf/func_sim.hpp"

#include "reef_perf/inst_decode.hpp"

#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace reef_perf {

FuncSim::FuncSim(const std::string& elf_path, const SpikeOptions& options)
    : driver_(elf_path, options) {}

InstPtr FuncSim::next() {
    if (halted_) {
        return nullptr;
    }
    std::optional<InstRecord> rec = driver_.step();
    if (!rec) {
        halted_ = true;
        return nullptr;
    }

    auto inst = std::make_shared<Inst>();
    inst->seq = executed_;
    inst->pc = rec->pc;
    inst->next_pc = rec->next_pc;
    inst->encoding = rec->encoding;
    inst->disasm = std::move(rec->disasm);
    inst->vl = rec->vl;
    inst->sew_bytes = rec->sew_bytes;
    inst->lmul8 = rec->lmul8;
    inst->mem = std::move(rec->mem);
    decode_inst(*inst);
    ++executed_;
    return inst;
}

} // namespace reef_perf
