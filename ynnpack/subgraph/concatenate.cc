// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "ynnpack/base/log.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/builder/pipeline.h"
#include "slinky/builder/simplify.h"
#include "slinky/runtime/expr.h"

namespace ynn {

extern "C" {

ynn_status ynn_define_concatenate(ynn_subgraph_t subgraph, int32_t axis,
                                  size_t num_inputs, const uint32_t* input_ids,
                                  uint32_t* output_id, uint32_t flags) {
  // Validate arguments.
  YNN_RETURN_IF_ERROR(validate_subgraph("concatenate", subgraph));
  YNN_RETURN_IF_ERROR(validate_input_tensor_array(
      "concatenate", subgraph, "input_ids", num_inputs, input_ids));
  YNN_RETURN_IF_ERROR(
      validate_output_tensor("concatenate", subgraph, "output_id", output_id));

  const ynn_value& input0 = subgraph->value(input_ids[0]);
  YNN_RETURN_IF_ERROR(
      validate_axis("concatenate", "input", input0.rank(), axis));
  axis = axis_to_slinky_dim(input0.rank(), axis);

  // Make the output and node.
  ynn_node node;

  std::vector<slinky::expr> output_extents = input0.extents;
  slinky::expr& extent_axis = output_extents[axis];
  if (!extent_axis.defined()) extent_axis = 1;
  for (int i = 1; i < num_inputs; ++i) {
    const ynn_value& input_i = subgraph->value(input_ids[i]);
    if (input0.rank() != input_i.rank()) {
      YNN_LOG_ERROR() << "For node `concatenate`, rank mismatch for input " << i
                      << " of concatenate";
      return ynn_status_invalid_parameter;
    }
    extent_axis += input_i.extent(axis);
    for (int d = 0; d < input0.rank(); ++d) {
      if (d == axis) continue;
      node.checks.push_back(
          {input_i.extent(d) == input0.extent(d),
           {"mismatch in non-concatenated dimension ", d, " of ",
            ynn_node::input_idx{0}, " (", input0.extent(d), ") and ",
            ynn_node::input_idx{i}, " (", input_i.extent(d), ")"}});
    }
  }
  extent_axis = slinky::simplify(extent_axis);

  ynn_value& output_value = subgraph->get_output_value(output_id, input0);
  output_value.extents = std::move(output_extents);

  node.inputs.assign(input_ids, input_ids + num_inputs);
  node.outputs = {*output_id};
  node.op = ynn_node::concatenate{axis};

  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const int32_t axis = std::get<ynn_node::concatenate>(node.op).axis;
    ynn_runtime_value& output = runtime.value(node.outputs[0]);
    const ynn_runtime_value& input_0 = runtime.value(node.inputs[0]);

    output.make_buffer(runtime, input_0.buffer->elem_size());

    std::vector<slinky::var> dims =
        runtime.globals.make_dims(output.buffer->rank());
    std::vector<slinky::buffer_expr_ptr> inputs;
    inputs.reserve(node.inputs.size());
    std::vector<slinky::expr> bounds = {0};
    for (uint32_t i : node.inputs) {
      const ynn_runtime_value& input_value = runtime.value(i);
      inputs.push_back(input_value.buffer);
      bounds.push_back(bounds.back() + input_value.extent(axis));
    }
    auto func =
        slinky::func::make_concat(inputs, {output.buffer, dims}, axis, bounds);
    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

}  // extern "C"

}  // namespace ynn
