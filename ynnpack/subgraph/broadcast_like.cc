// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <optional>
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

  // Propagate shape. Only the dimensions in `axes` are broadcasted. Dimensions
  // the input does not have are only added if they need to be broadcasted.
  const std::vector<slinky::expr>& template_extents = template_value.extents;
  std::vector<slinky::expr> output_extents = input.extents;
  for (int d = 0; d < static_cast<int>(template_extents.size()); ++d) {
    if (!axes_set[d]) continue;

    const slinky::expr& template_extent = template_extents[d];
    if (!template_extent.defined() || slinky::is_one(template_extent)) {
      continue;
    }

    if (d >= static_cast<int>(output_extents.size())) {
      output_extents.resize(d + 1);
    }
    slinky::expr& output_extent = output_extents[d];
    if (!output_extent.defined() || slinky::is_one(output_extent) ||
        slinky::prove_true(template_extent != 1)) {
      output_extent = template_extent;
      continue;
    }

    // Assuming extents are >= 1, `template_extent > output_extent` implies
    // `output_extent` must be 1, while `template_extent <= output_extent`
    // implies `template_extent` is either 1 or `output_extent` (a no-op).
    std::optional<bool> template_larger =
        slinky::attempt_to_prove(template_extent > output_extent);
    if (template_larger && *template_larger) {
      output_extent = template_extent;
    } else if (!template_larger) {
      output_extent = subgraph->globals.get(
          select(template_extent != 1, template_extent, output_extent), "b");
    }
  }

  if (*output_id != YNN_INVALID_VALUE_ID &&
      output_extents.size() < template_extents.size()) {
    // The output is already defined, it should have the rank of the template.
    output_extents.resize(template_extents.size());
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
