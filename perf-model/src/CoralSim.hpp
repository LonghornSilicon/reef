// Top-level Sparta simulation: builds the unit tree, wires the ports, and
// connects the functional simulator to Fetch.
//
// Tree (parameter paths are top.core.<unit>.params.<name>):
//
//   top.core.fetch     -> Fetch
//   top.core.dispatch  -> Dispatch
//   top.core.execute   -> Execute
//   top.core.rob       -> Rob

#pragma once

#include <memory>
#include <ostream>
#include <string>

#include "FuncSim.hpp"
#include "sparta/app/Simulation.hpp"

namespace coralnpu_perf {

class Fetch;
class Dispatch;
class Execute;
class Rob;

class CoralSim : public sparta::app::Simulation {
 public:
  CoralSim(sparta::Scheduler& scheduler, const std::string& elf_path,
           const cn_options_t& funcsim_options);
  ~CoralSim() override;

  // Human-readable end-of-run summary.
  void printSummary(std::ostream& os) const;
  // Same numbers as JSON, for scripts.
  void writeJson(std::ostream& os) const;

 private:
  void buildTree_() override;
  void configureTree_() override;
  void bindTree_() override;

  std::unique_ptr<FuncSim> funcsim_;
  std::string elf_path_;
  Fetch* fetch_ = nullptr;
  Dispatch* dispatch_ = nullptr;
  Execute* execute_ = nullptr;
  Rob* rob_ = nullptr;
};

}  // namespace coralnpu_perf
