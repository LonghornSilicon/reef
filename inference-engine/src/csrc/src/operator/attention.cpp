#include "inference_engine/operator/attention.hpp"

// The attention pipeline, weight and cache structs, and both mask policies
// are defined as templates in attention.hpp so any scalar type can
// instantiate them.

// @ Attention team
// TODO: Add explicit instantiations for common scalar types here, following
// tensor.cpp, once linear() lands and the first model pins its types.
