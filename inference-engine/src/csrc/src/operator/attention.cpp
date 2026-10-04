#include "inference_engine/operator/attention.hpp"

namespace inference_engine{

// @ Attention team
// TODO: Implement Attention's constructor, forward, project_qkv, and
// append_kv_cache. Keep template definitions header-visible or explicitly
// instantiate supported types. Match head counts and weight layout in
// workloads.

// @ Attention team
// TODO: Implement Attention::context and concrete masking policies, checking
// prefill/decode outputs and cached-token offsets against workloads.
    Tensor<float> Attention::mask_scores(const Tensor<float>& scores, std::size_t query_start) const {
        const std::vector<float>::const_iterator first = scores.values.begin() + std::min(query_start + 1, scores.values.size());
        std::fill(first, scores.values.end(), -std::numeric_limits<float>::infinity());
    }

// @ Attention team
// TODO: Implement GlobalAttention and LocalAttention constructors and masks,
// including sliding-window boundaries.

};