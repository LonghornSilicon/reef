#include "CoralSim.hpp"

#include <iomanip>

#include "Dispatch.hpp"
#include "Execute.hpp"
#include "Fetch.hpp"
#include "Rob.hpp"
#include "sparta/simulation/ResourceFactory.hpp"
#include "sparta/simulation/ResourceTreeNode.hpp"

namespace coralnpu_perf {

CoralSim::CoralSim(sparta::Scheduler& scheduler, const std::string& elf_path,
                   const cn_options_t& funcsim_options)
    : sparta::app::Simulation("coralnpu_perf", &scheduler),
      funcsim_(std::make_unique<FuncSim>(elf_path, funcsim_options)),
      elf_path_(elf_path) {
  auto* rs = getResourceSet();
  rs->addResourceFactory<sparta::ResourceFactory<Fetch, Fetch::FetchParameterSet>>();
  rs->addResourceFactory<sparta::ResourceFactory<Dispatch, Dispatch::DispatchParameterSet>>();
  rs->addResourceFactory<sparta::ResourceFactory<Execute, Execute::ExecuteParameterSet>>();
  rs->addResourceFactory<sparta::ResourceFactory<Rob, Rob::RobParameterSet>>();
}

CoralSim::~CoralSim() { getRoot()->enterTeardown(); }

void CoralSim::buildTree_() {
  auto* core = new sparta::TreeNode(getRoot(), "core", "CoralNPU scalar + vector core");
  to_delete_.emplace_back(core);

  for (const char* unit : {Fetch::name, Dispatch::name, Execute::name, Rob::name}) {
    auto* rtn = new sparta::ResourceTreeNode(core, unit, sparta::TreeNode::GROUP_NAME_NONE,
                                             sparta::TreeNode::GROUP_IDX_NONE, unit,
                                             getResourceSet()->getResourceFactory(unit));
    to_delete_.emplace_back(rtn);
  }
}

void CoralSim::configureTree_() {}

void CoralSim::bindTree_() {
  sparta::TreeNode* root = getRoot();
  auto port = [root](const std::string& path) {
    return root->getChildAs<sparta::Port>("core." + path);
  };

  sparta::bind(port("fetch.ports.out_insts"), port("dispatch.ports.in_insts"));
  sparta::bind(port("dispatch.ports.out_fetch_credits"), port("fetch.ports.in_credits"));
  sparta::bind(port("dispatch.ports.out_execute"), port("execute.ports.in_insts"));
  sparta::bind(port("dispatch.ports.out_rob"), port("rob.ports.in_insts"));
  sparta::bind(port("rob.ports.out_credits"), port("dispatch.ports.in_rob_credits"));
  sparta::bind(port("execute.ports.out_lsu_credits"), port("dispatch.ports.in_lsu_credits"));
  sparta::bind(port("execute.ports.out_vec_credits"), port("dispatch.ports.in_vec_credits"));

  auto unit = [root](const std::string& path) {
    return root->getChildAs<sparta::ResourceTreeNode>("core." + path);
  };
  fetch_ = unit("fetch")->getResourceAs<Fetch>();
  dispatch_ = unit("dispatch")->getResourceAs<Dispatch>();
  execute_ = unit("execute")->getResourceAs<Execute>();
  rob_ = unit("rob")->getResourceAs<Rob>();
  fetch_->setFuncSim(funcsim_.get());
}

void CoralSim::printSummary(std::ostream& os) const {
  const uint64_t cycles = rob_->lastRetireCycle();
  const uint64_t insts = rob_->numRetired();
  const double ipc = cycles ? static_cast<double>(insts) / cycles : 0.0;

  os << "\n==================== coralnpu_perf summary ====================\n"
     << "  workload            : " << elf_path_ << "\n"
     << "  instructions retired: " << insts << "\n"
     << "  cycles              : " << cycles << "\n"
     << "  IPC                 : " << std::fixed << std::setprecision(3) << ipc << "\n"
     << "  taken branches/jumps: " << fetch_->numRedirects() << "\n";

  if (insts != funcsim_->executed()) {
    os << "  WARNING: retired " << insts << " but MPACT executed " << funcsim_->executed()
       << " instructions (model deadlock?)\n";
  }

  os << "\n  Dispatch: cycles with no dispatch, by cause\n";
  const uint64_t busy = dispatch_->cyclesWithDispatch();
  os << "    " << std::left << std::setw(20) << "(dispatched)" << std::right
     << std::setw(12) << busy << "  " << std::setw(5) << std::setprecision(1)
     << (cycles ? 100.0 * busy / cycles : 0.0) << "%\n";
  for (size_t i = 0; i < static_cast<size_t>(StallReason::NUM_REASONS); ++i) {
    const auto r = static_cast<StallReason>(i);
    const uint64_t n = dispatch_->stallCycles(r);
    os << "    " << std::left << std::setw(20) << stallReasonName(r) << std::right
       << std::setw(12) << n << "  " << std::setw(5)
       << (cycles ? 100.0 * n / cycles : 0.0) << "%\n";
  }

  os << "\n  Execute: utilisation (busy cycles / (units * cycles))\n";
  for (const auto& s : execute_->poolStats()) {
    const double util = cycles ? 100.0 * s.busy_cycles / (static_cast<double>(s.count) * cycles) : 0.0;
    os << "    " << std::left << std::setw(8) << s.name << std::right << " x" << s.count
       << "  ops " << std::setw(10) << s.ops << "  util " << std::setw(5) << util << "%\n";
  }
  os << "================================================================\n";
}

void CoralSim::writeJson(std::ostream& os) const {
  const uint64_t cycles = rob_->lastRetireCycle();
  const uint64_t insts = rob_->numRetired();
  os << "{\n"
     << "  \"workload\": \"" << elf_path_ << "\",\n"
     << "  \"instructions\": " << insts << ",\n"
     << "  \"cycles\": " << cycles << ",\n"
     << "  \"ipc\": " << (cycles ? static_cast<double>(insts) / cycles : 0.0) << ",\n"
     << "  \"taken_redirects\": " << fetch_->numRedirects() << ",\n"
     << "  \"dispatch\": {\n"
     << "    \"cycles_with_dispatch\": " << dispatch_->cyclesWithDispatch();
  for (size_t i = 0; i < static_cast<size_t>(StallReason::NUM_REASONS); ++i) {
    const auto r = static_cast<StallReason>(i);
    os << ",\n    \"stall_" << stallReasonName(r) << "\": " << dispatch_->stallCycles(r);
  }
  os << "\n  },\n  \"pools\": {";
  bool first = true;
  for (const auto& s : execute_->poolStats()) {
    os << (first ? "\n" : ",\n") << "    \"" << s.name << "\": {\"units\": " << s.count
       << ", \"ops\": " << s.ops << ", \"busy_cycles\": " << s.busy_cycles << "}";
    first = false;
  }
  os << "\n  }\n}\n";
}

}  // namespace coralnpu_perf
