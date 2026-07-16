// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/builder/pipeline.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"

namespace ynn {

extern "C" {

ynn_status ynn_define_slice_like(ynn_subgraph_t subgraph, size_t num_axes,
                                 const int32_t* axes, uint32_t input_id,
                                 uint32_t template_id, uint32_t* output_id,
                                 uint32_t flags) {
  const bool keep_shape = (flags & YNN_NODE_FLAG_KEEP_SHAPE) != 0;

  // Validate arguments.
  YNN_RETURN_IF_ERROR(validate_subgraph("slice_like", subgraph));
  YNN_RETURN_IF_ERROR(
      validate_input_tensor("slice_like", subgraph, "input_id", input_id));
  YNN_RETURN_IF_ERROR(validate_input_tensor("slice_like", subgraph,
                                            "template_id", template_id));
  YNN_RETURN_IF_ERROR(
      validate_output_tensor("slice_like", subgraph, "output_id", output_id));
  const ynn_value& input = subgraph->value(input_id);
  const ynn_value& template_value = subgraph->value(template_id);

  ynn::axes_set axes_set;
  for (size_t i = 0; i < num_axes; ++i) {
    const int axis = axis_to_slinky_dim(input.rank(), axes[i]);
    if (axis >= 0 && axis < input.rank()) {
      axes_set[axis] = true;
    }
  }

  ynn_node node;
  node.inputs = {input_id, template_id};

  // Propagate shape.
  ynn_value& output = subgraph->get_output_value(output_id, input);
  output.extents = input.extents;
  if (!keep_shape) {
    for (int d = 0; d < output.extents.size(); ++d) {
      if (!axes_set[d]) continue;

      // We just want the intersection of the two buffers in this dimension.
      output.extents[d] = min(input.extent(d), template_value.extent(d));
    }
  }

  node.outputs = {*output_id};
  node.op = ynn_node::slice_like{axes_set, keep_shape};
  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const ynn_node::slice_like& op = std::get<ynn_node::slice_like>(node.op);
    const ynn_runtime_value& input = runtime.value(node.inputs[0]);
    const ynn_runtime_value& template_val = runtime.value(node.inputs[1]);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    if (op.keep_shape || output.is_external_output()) {
      output.make_buffer(runtime, input.buffer->elem_size());

      slinky::func func;
      const int rank = input.rank();
      std::vector<slinky::var> dims = runtime.globals.make_dims(rank);
      slinky::box_expr bounds =
          make_elementwise_bounds(dims, output.physical_extents());
      slinky::func::input func_input{input.buffer, bounds};

      if (op.keep_shape) {
        // If we are keeping the shape, we need to do a padded copy, and crop
        // the input to only the region we want to consume.
        func_input.input_crop.resize(rank);
        for (int d = 0; d < rank; ++d) {
          slinky::expr input_extent = input.physical_extent(d);
          slinky::expr extent =
              op.axes[d]
                  ? slinky::min(input_extent, template_val.physical_extent(d))
                  : input_extent;
          func_input.input_crop[d] = all_bounds(extent);
        }

        uint64_t zero = 0;
        const int elem_size = ynn::type_size_bytes(input.type);
        slinky::buffer_expr_ptr padding_val =
            slinky::buffer_expr::make_constant(
                runtime.globals.symbols, "zero",
                slinky::raw_buffer::make_scalar(elem_size, &zero));
        func = slinky::func::make_copy(std::move(func_input),
                                       {output.buffer, dims}, {padding_val});
      } else if (output.is_external_output()) {
        func = slinky::func::make_copy(std::move(func_input),
                                       {output.buffer, dims});
      }
      runtime.funcs.push_back(std::move(func));
    } else {
      // We don't actually need to do anything here, this op is only a
      // manipulation of the extents, slinky doesn't care about the extents of
      // the buffer (as long as they are big enough).
      output.buffer = input.buffer;
    }

    return ynn_status_success;
  };

  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

}  // extern "C"

}  // namespace ynn
