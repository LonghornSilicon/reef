// coralnpu_perf: coarse performance model of CoralNPU (M3).
//
//   coralnpu_perf --elf program.elf [-c configs/m3_coarse.yaml]
//                 [-p top.core.fetch.params.fetch_width 2] [--json out.json]
//
// All standard Sparta options also work (run with --help), for example
// --show-parameters, --write-final-config, --auto-summary on.

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "CoralSim.hpp"
#include "sparta/app/CommandLineSimulator.hpp"
#include "sparta/sparta.hpp"

namespace {

const char USAGE[] =
    "Usage:\n"
    "  coralnpu_perf --elf <program.elf> [options]\n"
    "\n"
    "Memory map (must match the RTL configuration the program was built for):\n"
    "  default : ITCM 0x0/8KB,  DTCM 0x10000/32KB\n"
    "  --highmem: ITCM 0x0/1MB, DTCM 0x100000/1MB\n";

uint32_t parseNumber(const std::string& s) {
  return static_cast<uint32_t>(std::stoul(s, nullptr, 0));
}

}  // namespace

int main(int argc, char** argv) {
  std::string elf;
  std::string json_path;
  std::string dtcm_range;

  sparta::app::DefaultValues defaults;
  defaults.auto_summary_default = "off";  // we print our own summary
  sparta::app::CommandLineSimulator cls(USAGE, defaults);

  auto& app_opts = cls.getApplicationOptions();
  app_opts.add_options()
      ("elf", sparta::app::named_value<std::string>("ELF", &elf),
       "RISC-V ELF to run (required)")
      ("json", sparta::app::named_value<std::string>("FILE", &json_path),
       "Write the summary as JSON to FILE")
      ("highmem", "Use the M3 highmem memory map (1 MB ITCM / 1 MB DTCM)",
       "Use the M3 highmem memory map")
      ("dtcm", sparta::app::named_value<std::string>("START:LEN", &dtcm_range),
       "Override the DTCM range, e.g. 0x10000:0x8000");

  int err = 0;
  if (!cls.parse(argc, argv, err)) return err;
  if (elf.empty()) {
    std::cerr << "error: --elf is required\n" << USAGE;
    return 2;
  }

  cn_options_t mem;
  cn_default_options(&mem);
  const auto& vm = cls.getVariablesMap();
  if (vm.count("highmem")) {
    mem.itcm_length = 1024 * 1024;
    mem.dtcm_start = 0x100000;
    mem.dtcm_length = 1024 * 1024;
  }
  if (!dtcm_range.empty()) {
    const std::string& s = dtcm_range;
    const size_t colon = s.find(':');
    if (colon == std::string::npos) {
      std::cerr << "error: --dtcm expects START:LEN\n";
      return 2;
    }
    mem.dtcm_start = parseNumber(s.substr(0, colon));
    mem.dtcm_length = parseNumber(s.substr(colon + 1));
  }

  sparta::Scheduler scheduler;
  coralnpu_perf::CoralSim sim(scheduler, elf, mem);
  cls.populateSimulation(&sim);
  cls.runSimulator(&sim);
  cls.postProcess(&sim);

  sim.printSummary(std::cout);
  if (!json_path.empty()) {
    std::ofstream out(json_path);
    sim.writeJson(out);
  }
  return 0;
}
