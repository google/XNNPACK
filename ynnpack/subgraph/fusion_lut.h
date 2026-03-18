#ifndef XNNPACK_YNNPACK_SUBGRAPH_FUSION_LUT_H_
#define XNNPACK_YNNPACK_SUBGRAPH_FUSION_LUT_H_

#include "ynnpack/subgraph/fusion_types.h"
#include "ynnpack/subgraph/subgraph.h"

namespace ynn {

// Rewrites a subgraph to use unary LUTs.
// Returns true if the subgraph was modified.
bool rewrite_subgraph_for_unary_lut(ynn_subgraph& subgraph,
                                    subgraph_analysis& analysis);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_FUSION_LUT_H_
