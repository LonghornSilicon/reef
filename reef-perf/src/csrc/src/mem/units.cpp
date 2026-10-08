#include "reef_perf/mem/axi.hpp"
#include "reef_perf/mem/tcm.hpp"

#include "sparta/utils/SpartaAssert.hpp"

#include <cstdint>
#include <span>

namespace reef_perf {

Tcm::Tcm(sparta::TreeNode* node, const TcmParameterSet* params)
    : sparta::Unit(node, name),
      itcm_("itcm", params->itcm_latency, params->itcm_line_bytes,
            params->itcm_cycles_per_line),
      dtcm_("dtcm", params->dtcm_latency, params->dtcm_line_bytes,
            params->dtcm_cycles_per_line) {}

MemResponse Tcm::access(RegionKind kind, std::span<const MemAccess> accesses,
                        std::uint64_t earliest) {
    sparta_assert(kind != RegionKind::EXT, "EXT accesses go to the AXI port");
    TcmModel& tcm = kind == RegionKind::ITCM ? itcm_ : dtcm_;
    return tcm.access(accesses, earliest);
}

Axi::Axi(sparta::TreeNode* node, const AxiParameterSet* params)
    : sparta::Unit(node, name),
      model_(params->axi_latency, params->axi_bytes_per_beat,
             params->axi_max_outstanding) {}

} // namespace reef_perf
