// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

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
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"

namespace ynn {

extern "C" {

ynn_status ynn_define_even_split(ynn_subgraph_t subgraph, int32_t axis,
                                 uint32_t input_id, size_t num_outputs,
                                 uint32_t* output_ids, uint32_t flags) {
  // Validate arguments.
  assert(subgraph);
  assert(subgraph->is_valid_value(input_id));
  for (size_t i = 0; i < num_outputs; ++i) {
    if (output_ids[i] == YNN_INVALID_VALUE_ID) {
      output_ids[i] =
          subgraph->new_internal_value(subgraph->value(input_id)).id;
    }
  }

  const ynn_value& input = subgraph->value(input_id);
  axis = ynn::axis_to_slinky_dim(input.rank(), axis);

  ynn_node node;
  node.inputs = {input_id};
  node.outputs.assign(output_ids, output_ids + num_outputs);
  node.op = ynn_node::even_split{axis};

  // Propagate shape.
  std::vector<slinky::expr> output_extents = input.extents;
  assert(output_extents[axis].defined());
  const int split_factor = num_outputs;

  node.checks.push_back({
      output_extents[axis] % split_factor == 0,
      {"In node 'even_split', invalid split by ", split_factor,
       " in dimension ", axis, " (", output_extents[axis], ") of ",
       ynn_node::input_idx{0}},
  });

  output_extents[axis] = output_extents[axis] / split_factor;

  for (size_t i = 0; i < num_outputs; ++i) {
    ynn_value& output = subgraph->value(output_ids[i]);
    output.extents = output_extents;
  }

  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const int32_t axis = std::get<ynn_node::even_split>(node.op).axis;
    const int split_factor = node.outputs.size();
    const ynn_runtime_value& input = runtime.value(node.inputs[0]);

    std::vector<slinky::var> dims =
        ynn::make_dims(input.buffer->rank(), runtime.symbols);

    slinky::expr delta = input.extents[axis] / split_factor;
    slinky::expr offset = 0;

    // We implement splits by making a func for each output.
    for (uint32_t i : node.outputs) {
      ynn_runtime_value& output = runtime.value(i);
      if (output.is_valid()) {
        output.make_buffer(runtime, input.buffer->elem_size());

        slinky::func::input func_input{
            input.buffer, ynn::make_elementwise_bounds(dims, input.extents)};
        func_input.bounds[axis] += offset;
        auto func = slinky::func::make_copy(std::move(func_input),
                                            {output.buffer, dims});

        auto sched = std::make_unique<ynn::scheduling_info>();
        sched->force_root = true;
        func.user_data() = sched.get();
        runtime.scheduling_info_storage.push_back(std::move(sched));
        runtime.funcs.push_back(std::move(func));
      }
      offset += delta;
    }
    return ynn_status_success;
  };
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

}  // extern "C"

}  // namespace ynn
