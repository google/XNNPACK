// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/static_transpose.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/runtime/expr.h"

namespace ynn {

extern "C" {

ynn_status ynn_define_broadcast(ynn_subgraph_t subgraph, size_t num_axes,
                                const int32_t* axes, uint32_t input_id,
                                uint32_t* output_id, uint32_t flags) {
  // Validate arguments.
  YNN_RETURN_IF_ERROR(validate_subgraph("broadcast", subgraph));
  YNN_RETURN_IF_ERROR(
      validate_input_tensor("broadcast", subgraph, "input_id", input_id));
  YNN_RETURN_IF_ERROR(
      validate_output_tensor("broadcast", subgraph, "output_id", output_id));
  const ynn_value& input = subgraph->value(input_id);

  ynn::axes_set axes_set;
  for (size_t i = 0; i < num_axes; ++i) {
    const int axis = axis_to_slinky_dim(input.rank(), axes[i]);
    if (axis < input.rank()) {
      // Dimensions past the input's rank are implicit broadcasts; broadcasting
      // a broadcast dimension is a no-op, so skip it.
      axes_set[axis] = true;
    }
  }

  ynn_node node;

  for (int d = 0; d < input.rank(); ++d) {
    if (!axes_set[d]) {
      // Not broadcasting this dimension.
      continue;
    }
    if (!input.extents[d].defined()) {
      // This dimension is already a broadcast.
      axes_set[d] = false;
      continue;
    }

    node.checks.push_back({
        input.extents[d] == 1,
        {"For node 'broadcast', invalid broadcast in dimension ", d, " of ",
         ynn_node::input_idx{0}},
    });
  }

  if (!axes_set.any() && *output_id == YNN_INVALID_VALUE_ID) {
    // This node is a no-op, skip it.
    *output_id = input_id;
    return ynn_status_success;
  }

  std::vector<int32_t> permutation(input.rank());
  for (size_t i = 0; i < permutation.size(); ++i) {
    if (axes_set[i]) {
      permutation[i] = input.rank();
    } else {
      permutation[i] = i;
    }
  }

  ynn::define_static_transpose(*subgraph, node, std::move(permutation),
                               input_id, output_id, /*alias=*/true);
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

}  // extern "C"

}  // namespace ynn
