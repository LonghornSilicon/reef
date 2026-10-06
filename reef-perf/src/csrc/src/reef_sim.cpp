#include "reef_perf/reef_sim.hpp"

#include "reef_perf/dispatch.hpp"
#include "reef_perf/execute.hpp"
#include "reef_perf/fetch.hpp"
#include "reef_perf/rob.hpp"

#include "sparta/simulation/ResourceFactory.hpp"
#include "sparta/simulation/ResourceTreeNode.hpp"

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <ostream>
#include <string>

namespace reef_perf {

namespace {

/** Percentage of part in whole, or 0 when whole is 0.
 *
 *  @param part Numerator.
 *  @param whole Denominator.
 *  @return 100 * part / whole.
 */
double percent(std::uint64_t part, std::uint64_t whole) {
    return whole == 0
               ? 0.0
               : 100.0 * static_cast<double>(part) / static_cast<double>(whole);
}

/** Instructions per cycle, or 0 when no cycles elapsed.
 *
 *  @param insts Instructions retired.
 *  @param cycles Cycles elapsed.
 *  @return insts / cycles.
 */
double ipc(std::uint64_t insts, std::uint64_t cycles) {
    return cycles == 0
               ? 0.0
               : static_cast<double>(insts) / static_cast<double>(cycles);
}

} // namespace

ReefSim::ReefSim(sparta::Scheduler& scheduler, const std::string& elf_path,
                 const SpikeOptions& options)
    : sparta::app::Simulation("reef_perf", &scheduler),
      funcsim_(std::make_unique<FuncSim>(elf_path, options)),
      elf_path_(elf_path) {
    auto* resources = getResourceSet();
    resources->addResourceFactory<
        sparta::ResourceFactory<Fetch, Fetch::FetchParameterSet>>();
    resources->addResourceFactory<
        sparta::ResourceFactory<Dispatch, Dispatch::DispatchParameterSet>>();
    resources->addResourceFactory<
        sparta::ResourceFactory<Execute, Execute::ExecuteParameterSet>>();
    resources->addResourceFactory<
        sparta::ResourceFactory<Rob, Rob::RobParameterSet>>();
}

ReefSim::~ReefSim() { getRoot()->enterTeardown(); }

void ReefSim::buildTree_() {
    auto core = std::make_unique<sparta::TreeNode>(getRoot(), "core",
                                                   "Reef scalar + vector core");
    for (const char* unit :
         {Fetch::name, Dispatch::name, Execute::name, Rob::name}) {
        to_delete_.emplace_back(std::make_unique<sparta::ResourceTreeNode>(
            core.get(), unit, sparta::TreeNode::GROUP_NAME_NONE,
            sparta::TreeNode::GROUP_IDX_NONE, unit,
            getResourceSet()->getResourceFactory(unit)));
    }
    to_delete_.emplace_back(std::move(core));
}

void ReefSim::configureTree_() {}

void ReefSim::bindTree_() {
    sparta::TreeNode* root = getRoot();
    const auto port = [root](const std::string& path) {
        return root->getChildAs<sparta::Port>("core." + path);
    };
    sparta::bind(port("fetch.ports.out_insts"),
                 port("dispatch.ports.in_insts"));
    sparta::bind(port("dispatch.ports.out_fetch_credits"),
                 port("fetch.ports.in_credits"));
    sparta::bind(port("dispatch.ports.out_execute"),
                 port("execute.ports.in_insts"));
    sparta::bind(port("dispatch.ports.out_rob"), port("rob.ports.in_insts"));
    sparta::bind(port("rob.ports.out_credits"),
                 port("dispatch.ports.in_rob_credits"));
    sparta::bind(port("execute.ports.out_lsu_credits"),
                 port("dispatch.ports.in_lsu_credits"));
    sparta::bind(port("execute.ports.out_vec_credits"),
                 port("dispatch.ports.in_vec_credits"));

    const auto unit = [root](const std::string& path) {
        return root->getChildAs<sparta::ResourceTreeNode>("core." + path);
    };
    fetch_ = unit("fetch")->getResourceAs<Fetch>();
    dispatch_ = unit("dispatch")->getResourceAs<Dispatch>();
    execute_ = unit("execute")->getResourceAs<Execute>();
    rob_ = unit("rob")->getResourceAs<Rob>();
    fetch_->set_func_sim(funcsim_.get());
}

void ReefSim::print_summary(std::ostream& os) const {
    const std::uint64_t cycles = rob_->last_retire_cycle();
    const std::uint64_t insts = rob_->num_retired();

    os << "\n==================== reef_perf summary ====================\n"
       << "  workload            : " << elf_path_ << "\n"
       << "  instructions retired: " << insts << "\n"
       << "  cycles              : " << cycles << "\n"
       << "  IPC                 : " << std::fixed << std::setprecision(3)
       << ipc(insts, cycles) << "\n"
       << "  taken branches/jumps: " << fetch_->num_redirects() << "\n";
    if (insts != funcsim_->executed()) {
        os << "  WARNING: retired " << insts << " but Spike executed "
           << funcsim_->executed() << " instructions (model deadlock?)\n";
    }

    os << "\n  Dispatch: cycles with no dispatch, by cause\n";
    os << std::setprecision(1);
    const std::uint64_t busy = dispatch_->cycles_with_dispatch();
    os << "    " << std::left << std::setw(20) << "(dispatched)" << std::right
       << std::setw(12) << busy << "  " << std::setw(5) << percent(busy, cycles)
       << "%\n";
    constexpr auto kReasons =
        static_cast<std::size_t>(StallReason::NUM_REASONS);
    for (std::size_t i = 0; i < kReasons; ++i) {
        const auto reason = static_cast<StallReason>(i);
        const std::uint64_t stalled = dispatch_->stall_cycles(reason);
        os << "    " << std::left << std::setw(20) << stall_reason_name(reason)
           << std::right << std::setw(12) << stalled << "  " << std::setw(5)
           << percent(stalled, cycles) << "%\n";
    }

    os << "\n  Execute: utilisation (busy cycles / (units * cycles))\n";
    for (const ResourcePool* pool : execute_->pools()) {
        os << "    " << std::left << std::setw(8) << pool->name() << std::right
           << " x" << pool->count() << "  ops " << std::setw(10) << pool->ops()
           << "  util " << std::setw(5)
           << percent(pool->busy_cycles(), pool->count() * cycles) << "%\n";
    }
    os << "============================================================\n";
}

void ReefSim::write_json(std::ostream& os) const {
    const std::uint64_t cycles = rob_->last_retire_cycle();
    const std::uint64_t insts = rob_->num_retired();
    os << "{\n"
       << R"(  "workload": ")" << elf_path_ << "\",\n"
       << "  \"instructions\": " << insts << ",\n"
       << "  \"functional_instructions\": " << funcsim_->executed() << ",\n"
       << "  \"cycles\": " << cycles << ",\n"
       << "  \"ipc\": " << ipc(insts, cycles) << ",\n"
       << "  \"taken_redirects\": " << fetch_->num_redirects() << ",\n"
       << "  \"dispatch\": {\n"
       << "    \"cycles_with_dispatch\": " << dispatch_->cycles_with_dispatch();
    constexpr auto kReasons =
        static_cast<std::size_t>(StallReason::NUM_REASONS);
    for (std::size_t i = 0; i < kReasons; ++i) {
        const auto reason = static_cast<StallReason>(i);
        os << ",\n    \"stall_" << stall_reason_name(reason)
           << "\": " << dispatch_->stall_cycles(reason);
    }
    os << "\n  },\n  \"pools\": {";
    bool first = true;
    for (const ResourcePool* pool : execute_->pools()) {
        os << (first ? "\n" : ",\n") << "    \"" << pool->name()
           << R"(": {"units": )" << pool->count()
           << ", \"ops\": " << pool->ops()
           << ", \"busy_cycles\": " << pool->busy_cycles() << "}";
        first = false;
    }
    os << "\n  }\n}\n";
}

} // namespace reef_perf
