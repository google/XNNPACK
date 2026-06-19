// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_STATIC_TRANSPOSE_H_
#define XNNPACK_YNNPACK_SUBGRAPH_STATIC_TRANSPOSE_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "ynnpack/subgraph/subgraph.h"

namespace ynn {

// Define a transpose node, optionally using a slinky copy that may alias even
// if dimension 0 is not stride 1 in the result.
void define_static_transpose(ynn_subgraph& subgraph, ynn_node& node,
                             std::vector<int32_t> permutation,
                             uint32_t input_id, uint32_t* output_id,
                             bool alias = false);

void define_static_expand_dims(ynn_subgraph& subgraph, ynn_node& node,
                               uint32_t input_id, uint32_t* output_id,
                               const axes_set& new_axes);

// If the given transpose is an expand_dims, returns the axes that were
// expanded.
std::optional<axes_set> get_static_expand_dims_axes(
    const ynn_node::static_transpose& op, int input_rank);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_STATIC_TRANSPOSE_H_
