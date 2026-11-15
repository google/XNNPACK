// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/copy.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/builder/simplify.h"
#include "slinky/runtime/expr.h"

namespace ynn {

extern "C" {

ynn_status ynn_define_broadcast_like(ynn_subgraph_t subgraph, size_t num_axes,
                                     const int32_t* axes, uint32_t input_id,
                                     uint32_t template_id, uint32_t* output_id,
                                     uint32_t flags) {
  // Validate arguments.
  YNN_RETURN_IF_ERROR(validate_subgraph("broadcast_like", subgraph));
  YNN_RETURN_IF_ERROR(
      validate_input_tensor("broadcast_like", subgraph, "input_id", input_id));
  YNN_RETURN_IF_ERROR(validate_input_tensor("broadcast_like", subgraph,
                                            "template_id", template_id));
  YNN_RETURN_IF_ERROR(validate_output_tensor("broadcast_like", subgraph,
                                             "output_id", output_id));
  const ynn_value& input = subgraph->value(input_id);
  const ynn_value& template_value = subgraph->value(template_id);
  ynn::axes_set axes_set;
  for (size_t i = 0; i < num_axes; ++i) {
    const int axis = axis_to_slinky_dim(input.rank(), axes[i]);
    if (axis >= 0 && axis < template_value.rank()) {
      axes_set[axis] = true;
    }
  }

  // Propagate shape. Any dimensions the template has that the input does not
  // are broadcasted.
  const std::vector<slinky::expr>& template_extents = template_value.extents;
  std::vector<slinky::expr> output_extents = input.extents;
  if (template_extents.size() > static_cast<size_t>(input.rank())) {
    output_extents.insert(output_extents.end(),
                          template_extents.begin() + input.rank(),
                          template_extents.end());
  }

  for (int d = 0; d < static_cast<int>(template_extents.size()); ++d) {
    if (!axes_set[d]) continue;

    const slinky::expr& template_extent = template_extents[d];
    slinky::expr& output_extent = output_extents[d];
    if (!template_extent.defined() || slinky::is_one(template_extent)) {
      continue;
    }

    if (!output_extent.defined() || slinky::is_one(output_extent)) {
      output_extent = template_extent;
    } else if (!slinky::prove_true(template_extent == output_extent)) {
      output_extent = subgraph->globals.get(
          select(template_extent != 1, template_extent, output_extent), "b");
    }
  }

  ynn_node node;
  define_static_broadcast(*subgraph, node, std::move(output_extents), input_id,
                          output_id);
  if (node.is_valid()) {
    subgraph->add_node(std::move(node));
  }
  return ynn_status_success;
}

}  // extern "C"

}  // namespace ynn
