#include "reef_perf/reef_sim.hpp"

#include "reef_perf/common/resource_pool.hpp"

#include "sparta/ports/Port.hpp"

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

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
    mem_.set_memory_map(options.regions);
    for (Module* module : modules()) {
        module->add_factories(*getResourceSet());
    }
}

ReefSim::~ReefSim() { getRoot()->enterTeardown(); }

std::vector<Module*> ReefSim::modules() {
    return {&frontend_, &backend_, &vector_, &mem_};
}

std::vector<const Module*> ReefSim::modules() const {
    return {&frontend_, &backend_, &vector_, &mem_};
}

void ReefSim::buildTree_() {
    for (Module* module : modules()) {
        module->build(getRoot(), *getResourceSet(), to_delete_);
    }
}

void ReefSim::configureTree_() {}

void ReefSim::bind_ports(const Module& from, const char* from_port,
                         const Module& to, const char* to_port) {
    sparta::TreeNode* root = getRoot();
    sparta::bind(root->getChildAs<sparta::Port>(from.port_path(from_port)),
                 root->getChildAs<sparta::Port>(to.port_path(to_port)));
}

void ReefSim::bindTree_() {
    for (Module* module : modules()) {
        module->bind();
    }

    // The interfaces between modules (docs/interfaces.md).
    bind_ports(frontend_, Frontend::kOutInsts, backend_, Backend::kInInsts);
    bind_ports(backend_, Backend::kOutFetchCredits, frontend_,
               Frontend::kInCredits);
    bind_ports(backend_, Backend::kOutVector, vector_, Vector::kInInsts);
    bind_ports(vector_, Vector::kOutCredits, backend_,
               Backend::kInVectorCredits);

    frontend_.set_func_sim(funcsim_.get());
    backend_.set_memory(&mem_);
}

void ReefSim::print_summary(std::ostream& os) const {
    const std::uint64_t cycles = backend_.last_retire_cycle();
    const std::uint64_t insts = backend_.num_retired();
    os << "\n==================== reef_perf summary ====================\n"
       << "  workload            : " << elf_path_ << "\n"
       << "  instructions retired: " << insts << "\n"
       << "  cycles              : " << cycles << "\n"
       << "  IPC                 : " << std::fixed << std::setprecision(3)
       << ipc(insts, cycles) << "\n"
       << "  taken branches/jumps: " << frontend_.num_redirects() << "\n";
    if (insts != funcsim_->executed()) {
        os << "  WARNING: retired " << insts << " but Spike executed "
           << funcsim_->executed() << " instructions (model deadlock?)\n";
    }

    os << "\n  Dispatch: cycles with no dispatch, by cause\n";
    os << std::setprecision(1);
    const std::uint64_t busy = backend_.cycles_with_dispatch();
    os << "    " << std::left << std::setw(20) << "(dispatched)" << std::right
       << std::setw(12) << busy << "  " << std::setw(5) << percent(busy, cycles)
       << "%\n";
    constexpr auto kReasons =
        static_cast<std::size_t>(StallReason::NUM_REASONS);
    for (std::size_t i = 0; i < kReasons; ++i) {
        const auto reason = static_cast<StallReason>(i);
        const std::uint64_t stalled = backend_.stall_cycles(reason);
        os << "    " << std::left << std::setw(20) << stall_reason_name(reason)
           << std::right << std::setw(12) << stalled << "  " << std::setw(5)
           << percent(stalled, cycles) << "%\n";
    }

    os << "\n  Pools: utilisation (busy cycles / (units * cycles))\n";
    for (const Module* module : modules()) {
        for (const ResourcePool* pool : module->pools()) {
            os << "    " << std::left << std::setw(16)
               << (module->name() + "." + pool->name()) << std::right << " x"
               << pool->count() << "  ops " << std::setw(10) << pool->ops()
               << "  util " << std::setw(5)
               << percent(pool->busy_cycles(), pool->count() * cycles)
               << "%\n";
        }
    }
    os << "============================================================\n";
}

void ReefSim::write_json(std::ostream& os) const {
    const std::uint64_t cycles = backend_.last_retire_cycle();
    const std::uint64_t insts = backend_.num_retired();
    os << "{\n"
       << R"(  "workload": ")" << elf_path_ << "\",\n"
       << "  \"instructions\": " << insts << ",\n"
       << "  \"functional_instructions\": " << funcsim_->executed() << ",\n"
       << "  \"cycles\": " << cycles << ",\n"
       << "  \"ipc\": " << ipc(insts, cycles) << ",\n"
       << "  \"taken_redirects\": " << frontend_.num_redirects() << ",\n"
       << "  \"dispatch\": {\n"
       << "    \"cycles_with_dispatch\": " << backend_.cycles_with_dispatch();
    constexpr auto kReasons =
        static_cast<std::size_t>(StallReason::NUM_REASONS);
    for (std::size_t i = 0; i < kReasons; ++i) {
        const auto reason = static_cast<StallReason>(i);
        os << ",\n    \"stall_" << stall_reason_name(reason)
           << "\": " << backend_.stall_cycles(reason);
    }
    // Pool names are unique across modules (docs/interfaces.md), so the JSON
    // keeps its flat {"alu": ..., "vector": ...} layout.
    os << "\n  },\n  \"pools\": {";
    bool first = true;
    for (const Module* module : modules()) {
        for (const ResourcePool* pool : module->pools()) {
            os << (first ? "\n" : ",\n") << "    \"" << pool->name()
               << R"(": {"units": )" << pool->count()
               << ", \"ops\": " << pool->ops()
               << ", \"busy_cycles\": " << pool->busy_cycles() << "}";
            first = false;
        }
    }
    os << "\n  }\n}\n";
}

} // namespace reef_perf
