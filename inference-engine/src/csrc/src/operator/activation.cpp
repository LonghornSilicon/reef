#include "inference_engine/operator/activation.hpp"

// masked_score and the dimension-aware softmax are implemented as templates in
// activation.hpp so any scalar type can instantiate them; gelu is declared
// there and still needs an implementation. This translation unit only checks
// that the header compiles on its own.
