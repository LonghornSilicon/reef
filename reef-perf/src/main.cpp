//===- main.cpp - The reef_perf simulator ---------------------------------===//
///
/// \file
/// reef_perf: coarse performance model of the Reef NPU.
///
///   reef_perf --elf program.elf [-c configs/m3.yaml]
///             [-p top.frontend.params.fetch_width 2] [--json out.json]
///
/// All standard Sparta options also work (run with --help), for example
/// --show-parameters, --write-final-config and --auto-summary on.
///
//===----------------------------------------------------------------------===//

#include "ReefSim.h"
#include "frontend/FuncSim.h"

#include "sparta/app/CommandLineSimulator.hpp"
#include "sparta/sparta.hpp"

#include <exception>
#include <fstream>
#include <iostream>
#include <string>

using namespace reef_perf;

static constexpr const char *Usage =
    "Usage:\n"
    "  reef_perf --elf <program.elf> [options]\n"
    "\n"
    "Memory map (must match the RTL configuration the program was built "
    "for):\n"
    "  default  : ITCM 0x0/8KB,  DTCM 0x10000/32KB\n"
    "  --highmem: ITCM 0x0/1MB,  DTCM 0x100000/1MB\n";

namespace {

/// Command-line options specific to reef_perf.
struct Options {
  std::string Elf;              // ELF to run.
  std::string JSONPath;         // Optional JSON summary path.
  std::string ISA = DefaultISA; // ISA string for Spike.
};

} // namespace

/// Parses the command line and runs the simulation. \returns the exit code.
static int run(int Argc, char **Argv) {
  Options Opts;
  sparta::app::DefaultValues Defaults;
  Defaults.auto_summary_default = "off"; // We print our own summary.
  sparta::app::CommandLineSimulator CLI(Usage, Defaults);

  auto &AppOpts = CLI.getApplicationOptions();
  AppOpts.add_options()("elf",
                        sparta::app::named_value<std::string>("ELF", &Opts.Elf),
                        "RISC-V ELF to run (required)")(
      "json", sparta::app::named_value<std::string>("FILE", &Opts.JSONPath),
      "Write the summary as JSON to FILE")(
      "isa", sparta::app::named_value<std::string>("ISA", &Opts.ISA),
      "Spike ISA string (default: Reef M3)")(
      "highmem", "Use the M3 highmem memory map (1 MB ITCM / 1 MB DTCM)",
      "Use the M3 highmem memory map");

  int Err = 0;
  if (!CLI.parse(Argc, Argv, Err))
    return Err;
  if (Opts.Elf.empty()) {
    std::cerr << "error: --elf is required\n" << Usage;
    return 2;
  }

  SpikeOptions Spike;
  Spike.ISA = Opts.ISA;
  if (CLI.getVariablesMap().contains("highmem"))
    Spike.Regions = getHighmemMemoryMap();

  sparta::Scheduler Scheduler;
  ReefSim Sim(Scheduler, Opts.Elf, Spike);
  CLI.populateSimulation(&Sim);
  CLI.runSimulator(&Sim);
  CLI.postProcess(&Sim);

  Sim.printSummary(std::cout);
  if (!Opts.JSONPath.empty()) {
    std::ofstream Out(Opts.JSONPath);
    Sim.writeJSON(Out);
  }
  return 0;
}

int main(int argc, char **argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception &E) {
    std::cerr << "reef_perf: " << E.what() << "\n";
    return 1;
  }
}
