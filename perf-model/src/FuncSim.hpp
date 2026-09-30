// Functional simulator: runs the program in MPACT, one instruction at a time.
//
// Fetch calls next() for every instruction it fetches (execute-at-fetch).
// MPACT executes the instruction immediately; next() returns an Inst that
// already knows its registers, memory addresses and branch outcome.
//
// CoralNPU is in-order and never executes wrong-path instructions, so MPACT
// never has to be rolled back: the timing model only decides *when* each
// instruction happens, never *whether* it happens.

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "Inst.hpp"
#include "coralnpu_driver.h"

namespace coralnpu_perf {

class FuncSim {
 public:
  FuncSim(const std::string& elf_path, const cn_options_t& options);
  ~FuncSim();

  FuncSim(const FuncSim&) = delete;
  FuncSim& operator=(const FuncSim&) = delete;

  // Executes the next instruction. Returns nullptr once the program halts.
  InstPtr next();

  bool halted() const { return halted_; }
  uint64_t executed() const { return executed_; }

 private:
  void* handle_ = nullptr;
  std::unique_ptr<cn_inst_t> rec_;  // large (~8 KB); reused for every step
  bool halted_ = false;
  uint64_t executed_ = 0;
};

}  // namespace coralnpu_perf
