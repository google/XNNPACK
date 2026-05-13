// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_STATIC_TRANSPOSE_H_
#define XNNPACK_YNNPACK_SUBGRAPH_STATIC_TRANSPOSE_H_

#include <cstdint>
#include <vector>

#include "ynnpack/subgraph/subgraph.h"

namespace ynn {

// Define a transpose node, optionally using a slinky copy that may alias even
// if dimension 0 is not stride 1 in the result.
void define_static_transpose(ynn_subgraph& subgraph, ynn_node& node,
                             std::vector<int32_t> permutation,
                             uint32_t input_id, uint32_t& output_id,
                             bool alias = false);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_STATIC_TRANSPOSE_H_
