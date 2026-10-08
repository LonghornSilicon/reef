#include "inference_engine/operator/attention.hpp"

// The attention pipeline, the weight and cache structs, and both mask policies
// are implemented as templates in attention.hpp so any scalar type can
// instantiate them. This translation unit only checks that the header compiles
// on its own; explicit instantiations belong here once linear() lands and the
// first model pins its scalar types.
