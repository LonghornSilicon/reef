// Prints the dynamic instruction stream of an ELF as Spike executes it.
//
//   reef_trace <elf> [max_instructions]
//
// One line per instruction: sequence number, pc, encoding, next pc,
// vl/SEW/LMUL, decoded class, number of memory accesses and disassembly.
// Useful for checking a workload before running the timing model.

#include "reef_perf/frontend/func_sim.hpp"
#include "reef_perf/common/inst.hpp"
#include "reef_perf/frontend/spike_driver.hpp"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <string>

namespace {

/** Prints the trace.
 *
 *  @param argc Argument count.
 *  @param argv Argument vector.
 *  @return Process exit code.
 */
int run(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <elf> [max_instructions]\n", argv[0]);
        return 2;
    }
    const std::uint64_t max_insts =
        argc > 2 ? std::stoull(argv[2], nullptr, 0)
                 : std::numeric_limits<std::uint64_t>::max();

    reef_perf::FuncSim funcsim(argv[1], reef_perf::SpikeOptions{});
    std::uint64_t count = 0;
    while (count < max_insts) {
        const reef_perf::InstPtr inst = funcsim.next();
        if (!inst) {
            break;
        }
        std::printf("%6llu %08x %08x -> %08x vl=%u e%u m%u/8 %-11s mem=%zu  "
                    "%s\n",
                    static_cast<unsigned long long>(inst->seq), inst->pc,
                    inst->encoding, inst->next_pc, inst->vl,
                    inst->sew_bytes * 8U, inst->lmul8,
                    reef_perf::class_name(inst->cls), inst->mem.size(),
                    inst->disasm.c_str());
        ++count;
    }
    std::printf("# %llu instructions, %s\n",
                static_cast<unsigned long long>(count),
                funcsim.halted() ? "halted" : "limit reached");
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "reef_trace: %s\n", e.what());
        return 1;
    }
}
