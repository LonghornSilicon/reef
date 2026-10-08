// reef_perf: coarse performance model of the Reef NPU.
//
//   reef_perf --elf program.elf [-c configs/m3.yaml]
//             [-p top.frontend.fetch.params.fetch_width 2] [--json out.json]
//
// All standard Sparta options also work (run with --help), for example
// --show-parameters, --write-final-config and --auto-summary on.

#include "reef_perf/frontend/spike_driver.hpp"
#include "reef_perf/reef_sim.hpp"

#include "sparta/app/CommandLineSimulator.hpp"
#include "sparta/sparta.hpp"

#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>

namespace {

constexpr const char* kUsage =
    "Usage:\n"
    "  reef_perf --elf <program.elf> [options]\n"
    "\n"
    "Memory map (must match the RTL configuration the program was built "
    "for):\n"
    "  default  : ITCM 0x0/8KB,  DTCM 0x10000/32KB\n"
    "  --highmem: ITCM 0x0/1MB,  DTCM 0x100000/1MB\n";

/// Command-line options specific to reef_perf.
struct Options {
    /// ELF to run.
    std::string elf;
    /// Optional JSON summary path.
    std::string json_path;
    /// ISA string for Spike.
    std::string isa = reef_perf::kDefaultIsa;
};

/** Parses the command line and runs the simulation.
 *
 *  @param argc Argument count.
 *  @param argv Argument vector.
 *  @return Process exit code.
 */
int run(int argc, char** argv) {
    Options opts;
    sparta::app::DefaultValues defaults;
    defaults.auto_summary_default = "off"; // we print our own summary
    sparta::app::CommandLineSimulator cls(kUsage, defaults);

    auto& app_opts = cls.getApplicationOptions();
    app_opts.add_options()(
        "elf", sparta::app::named_value<std::string>("ELF", &opts.elf),
        "RISC-V ELF to run (required)")(
        "json", sparta::app::named_value<std::string>("FILE", &opts.json_path),
        "Write the summary as JSON to FILE")(
        "isa", sparta::app::named_value<std::string>("ISA", &opts.isa),
        "Spike ISA string (default: Reef M3)")(
        "highmem", "Use the M3 highmem memory map (1 MB ITCM / 1 MB DTCM)",
        "Use the M3 highmem memory map");

    int err = 0;
    if (!cls.parse(argc, argv, err)) {
        return err;
    }
    if (opts.elf.empty()) {
        std::cerr << "error: --elf is required\n" << kUsage;
        return 2;
    }

    reef_perf::SpikeOptions spike;
    spike.isa = opts.isa;
    if (cls.getVariablesMap().contains("highmem")) {
        spike.regions = reef_perf::highmem_memory_map();
    }

    sparta::Scheduler scheduler;
    reef_perf::ReefSim sim(scheduler, opts.elf, spike);
    cls.populateSimulation(&sim);
    cls.runSimulator(&sim);
    cls.postProcess(&sim);

    sim.print_summary(std::cout);
    if (!opts.json_path.empty()) {
        std::ofstream out(opts.json_path);
        sim.write_json(out);
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "reef_perf: " << e.what() << "\n";
        return 1;
    }
}
