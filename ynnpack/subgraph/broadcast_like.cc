// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/builder/pipeline.h"
#include "slinky/builder/simplify.h"
#include "slinky/runtime/expr.h"

namespace ynn {

extern "C" {

ynn_status ynn_define_broadcast_like(ynn_subgraph_t subgraph, size_t num_axes,
                                     const int32_t* axes, uint32_t input_id,
                                     uint32_t template_id, uint32_t* output_id,
                                     uint32_t flags) {
  // Validate arguments.
  assert(subgraph);
  assert(subgraph->is_valid_value(input_id));
  assert(subgraph->is_valid_value(template_id));
  assert(output_id);
  const ynn_value& input = subgraph->value(input_id);
  const ynn_value& template_value = subgraph->value(template_id);
  ynn::axes_set axes_set;
  for (size_t i = 0; i < num_axes; ++i) {
    const int axis = axis_to_slinky_dim(input.rank(), axes[i]);
    if (axis < template_value.rank()) {
      axes_set[axis] = true;
    }
  }

  ynn_node node;

  // Propagate shape.

  // Any dimensions the template has that the input does not are broadcasted.
  const std::vector<slinky::expr>& template_extents = template_value.extents;
  std::vector<slinky::expr> output_extents = input.extents;
  if (template_extents.size() > input.rank()) {
    output_extents.insert(output_extents.end(),
                          template_extents.begin() + input.rank(),
                          template_extents.end());
  }

  for (int d = 0; d < template_extents.size(); ++d) {
    if (!axes_set[d]) {
      // Not broadcasting this dimension.
      continue;
    }

    const slinky::expr& template_extent = template_extents[d];
    slinky::expr& output_extent = output_extents[d];

    if (!template_extent.defined() ||
        slinky::prove_true(template_extent == output_extent) ||
        slinky::is_one(template_extent)) {
      // This broadcast is a no-op in this dimension.
      axes_set[d] = false;
      continue;
    }

    // Implement broadcasting logic: if the input is 1, we take the
    // extent of the other operand.
    if (!output_extent.defined() || slinky::is_one(output_extent)) {
      // We know the output extent is one, just use the template.
      output_extent = template_extent;
    } else {
      output_extent = subgraph->make_global_variable(
          select(template_extent > 1, template_extent, output_extent), "b");

      node.checks.push_back({
          template_extent == 1 || output_extent == template_extent,
          {"invalid broadcast in dimension ", d, " of ", ynn_node::input_idx{0},
           " and ", ynn_node::input_idx{1}},
      });
    }
  }

  if (!axes_set.any() && *output_id == YNN_INVALID_VALUE_ID) {
    // This node is a no-op, skip it.
    *output_id = input_id;
    return ynn_status_success;
  }

  ynn_value& output = subgraph->get_output_value(output_id, input);
  output.extents = std::move(output_extents);

  // Note the template is not an input to the node, we only needed its shape.
  node.inputs = {input_id, template_id};
  node.outputs = {*output_id};
  node.op = ynn_node::broadcast_like{axes_set};
  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const ynn::axes_set& axes =
        std::get<ynn_node::broadcast_like>(node.op).axes;
    const int input_id = node.inputs[0];
    const int output_id = node.outputs[0];
    const ynn_runtime_value& input = runtime.value(input_id);
    ynn_runtime_value& output = runtime.value(output_id);

    if (!axes.any() && !output.is_external_output()) {
      // This node is a no-op, skip it.
      output.buffer = input.buffer;
      return ynn_status_success;
    }

    output.make_buffer(runtime, input.buffer->elem_size());

    std::vector<slinky::var> dims = make_dims(output.rank(), runtime.symbols);
    slinky::box_expr bounds = make_elementwise_bounds(dims, input.extents);

    for (size_t i = 0; i < axes.size(); ++i) {
      if (!axes[i]) continue;
      bounds[i] =
          make_broadcast_bounds(dims[i], input.extents[i], output.extents[i]);
    }

    if (bounds.size() > input.rank()) {
      bounds.resize(input.rank());
    }

    auto func = slinky::func::make_copy({input.buffer, std::move(bounds)},
                                        {output.buffer, std::move(dims)});
    runtime.funcs.push_back(std::move(func));

    auto sched = std::make_unique<scheduling_info>();

    // Schedule the output buffer to be stored at the same level it's
    // computed at.
    scheduled_buffer sched_bc_buffer = {output.buffer, 0};
    sched->scheduled_buffers.push_back(std::move(sched_bc_buffer));

    runtime.funcs.back().user_data() = sched.get();
    runtime.scheduling_info_storage.push_back(std::move(sched));

    return ynn_status_success;
  };
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

}  // extern "C"

}  // namespace ynn
