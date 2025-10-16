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
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"

namespace ynn {

extern "C" {

ynn_status ynn_define_static_pad(ynn_subgraph_t subgraph, size_t num_axes,
                                 const int32_t* axes,
                                 const int64_t* pre_paddings,
                                 const int64_t* post_paddings,
                                 uint32_t input_id, uint32_t padding_id,
                                 uint32_t* output_id, uint32_t flags) {
  // Validate arguments.
  assert(subgraph);
  assert(subgraph->is_valid_value(input_id));
  assert(output_id);
  const ynn_value& input = subgraph->value(input_id);

  ynn_node::static_pad op;
  op.paddings.reserve(num_axes);
  for (size_t i = 0; i < num_axes; ++i) {
    op.paddings.push_back({ynn::axis_to_slinky_dim(input.rank(), axes[i]),
                           pre_paddings[i], post_paddings[i]});
  }

  // Propagate shape.
  ynn_value& output = subgraph->get_output_value(output_id, input);
  output.extents = input.extents;

  for (const ynn_node::static_pad::padding& p : op.paddings) {
    output.extents[p.axis] += static_cast<slinky::index_t>(p.pre_padding) +
                              static_cast<slinky::index_t>(p.post_padding);
  }

  ynn_node node;
  node.inputs = {input_id, padding_id};
  node.outputs = {*output_id};
  node.op = std::move(op);
  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const ynn_node::static_pad& op = std::get<ynn_node::static_pad>(node.op);
    const ynn_runtime_value& input = runtime.value(node.inputs[0]);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    const int rank = output.rank();
    std::vector<slinky::var> dims = make_dims(rank, runtime.symbols);

    output.make_buffer(runtime, input.buffer->elem_size());

    slinky::func::input func_input{
        input.buffer, make_elementwise_bounds(dims, input.extents)};
    func_input.input_crop.resize(rank);
    for (const ynn_node::static_pad::padding& p : op.paddings) {
      func_input.bounds[p.axis] -= p.pre_padding;
      if (input.extents[p.axis].defined()) {
        func_input.input_crop[p.axis] =
            slinky::min_extent(0, input.extents[p.axis]);
      }
    }

    if (node.inputs[1] != YNN_INVALID_VALUE_ID) {
      const ynn_runtime_value& padding_value = runtime.value(node.inputs[1]);
      slinky::func::input padding{
          padding_value.buffer,
          make_elementwise_bounds(dims, padding_value.extents)};
      for (const ynn_node::static_pad::padding& p : op.paddings) {
        padding.bounds[p.axis] -= p.pre_padding;
      }

      auto func = slinky::func::make_copy(std::move(func_input),
                                          {output.buffer, std::move(dims)},
                                          std::move(padding));
      runtime.funcs.push_back(std::move(func));
    } else {
      auto func = slinky::func::make_copy(std::move(func_input),
                                          {output.buffer, std::move(dims)});
      runtime.funcs.push_back(std::move(func));
    }
    return ynn_status_success;
  };
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

}  // extern "C"

}  // namespace ynn
