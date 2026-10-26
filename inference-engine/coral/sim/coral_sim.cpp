// Runs a Coral ELF on Spike with no timing model:
//
//   coral_sim [--highmem] --elf ELF
//
// This is reef_perf's functional half, reef-perf's FuncSim: the same ISA,
// memory map and mpause handling, without Sparta. It is enough to tell
// whether a golden test passed, and it builds in minutes rather than hours,
// so CI uses it (coral/tools/setup_sim.sh builds it). Exits 0 when the
// program reaches mpause and 1 if it traps.
#include "common/MemoryMap.h"
#include "frontend/FuncSim.h"

#include <cstddef>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    reef_perf::SpikeOptions options;
    std::string elf;
    for (std::size_t index = 0; index < args.size(); ++index) {
        if (args[index] == "--highmem") {
            options.Regions = reef_perf::getHighmemMemoryMap();
        } else if (args[index] == "--elf" && index + 1 < args.size()) {
            elf = args[++index];
        } else {
            std::cerr << "usage: coral_sim [--highmem] --elf ELF\n";
            return 2;
        }
    }
    if (elf.empty()) {
        std::cerr << "usage: coral_sim [--highmem] --elf ELF\n";
        return 2;
    }

    try {
        reef_perf::FuncSim sim(elf, options);
        while (sim.next()) {
        }
        std::cout << "  instructions retired: " << sim.getNumExecuted() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "coral_sim: " << error.what() << '\n';
        return 1;
    }
}
