#include "inference_engine/operator/attention.hpp"

// forward and context are defined as templates in attention.hpp.

// @ Attention team
// TODO: Implement Attention's constructor, project_qkv, and append_kv_cache.
// Match head counts and weight layout in workloads. Once the weight and
// cache structs are defined, add explicit instantiations for common scalar
// types here, following tensor.cpp.

// @ Attention team
// TODO: Implement GlobalAttention and LocalAttention constructors and masks,
// including sliding-window boundaries.