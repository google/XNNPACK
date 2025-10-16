// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/builder/pipeline.h"
#include "slinky/runtime/expr.h"

namespace ynn {

extern "C" {

ynn_status ynn_define_stack(ynn_subgraph_t subgraph, int32_t axis,
                            size_t num_inputs, const uint32_t* input_ids,
                            uint32_t* output_id, uint32_t flags) {
  // Validate arguments.
  assert(subgraph);
  assert(input_ids);
  assert(output_id);
  assert(subgraph->is_valid_value(input_ids[0]));
  const ynn_value& input0 = subgraph->value(input_ids[0]);
  axis = axis_to_slinky_dim(input0.rank() + 1, axis);

  // Make the output and node.
  ynn_node node;

  for (int i = 1; i < num_inputs; ++i) {
    assert(subgraph->is_valid_value(input_ids[i]));
    const ynn_value& input_i = subgraph->value(input_ids[i]);
    assert(input0.rank() == input_i.rank());
    for (int d = 0; d < input0.rank(); ++d) {
      node.checks.push_back(
          {input_i.extents[d] == input0.extents[d],
           {"mismatch in dimension ", d, " of ", ynn_node::input_idx{0}, " (",
            input0.extents[d], ") and ", ynn_node::input_idx{i}, " (",
            input_i.extents[d], ")"}});
    }
  }

  ynn_value& output_value = subgraph->get_output_value(output_id, input0);
  output_value.extents = input0.extents;
  if (output_value.extents.size() < axis) {
    output_value.extents.resize(axis);
  }
  output_value.extents.insert(output_value.extents.begin() + axis, num_inputs);

  node.inputs.assign(input_ids, input_ids + num_inputs);
  node.outputs = {*output_id};
  node.op = ynn_node::stack{axis};

  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const int32_t axis = std::get<ynn_node::stack>(node.op).axis;
    ynn_runtime_value& output = runtime.value(node.outputs[0]);
    const ynn_runtime_value& input_0 = runtime.value(node.inputs[0]);

    output.make_buffer(runtime, input_0.buffer->elem_size());

    std::vector<slinky::var> dims =
        make_dims(output.buffer->rank(), runtime.symbols);
    std::vector<slinky::buffer_expr_ptr> inputs;
    inputs.reserve(node.inputs.size());
    for (uint32_t i : node.inputs) {
      const ynn_runtime_value& input_value = runtime.value(i);
      inputs.push_back(input_value.buffer);
    }
    auto func = slinky::func::make_stack(inputs, {output.buffer, dims}, axis);
    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

}  // extern "C"

}  // namespace ynn
