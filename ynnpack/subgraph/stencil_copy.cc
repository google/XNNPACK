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
#include "slinky/base/arithmetic.h"
#include "slinky/builder/pipeline.h"
#include "slinky/builder/simplify.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"

namespace ynn {

namespace {

int dilated_kernel_size(const ynn_node::stencil_copy::stencil& stencil) {
  return (stencil.extent - 1) * stencil.dilation + 1;
}

template <typename T>
T compute_same_padding_min(const ynn_node::stencil_copy::stencil& stencil,
                           T input_extent) {
  T output_extent = slinky::ceil_div<T>(input_extent, stencil.stride);
  T unpadded_extent =
      (output_extent - 1) * stencil.stride + dilated_kernel_size(stencil);
  return max(unpadded_extent - input_extent, 0) / 2;
}

}  // namespace

extern "C" {

using stencil_info = ynn_node::stencil_copy::stencil;

ynn_status ynn_define_stencil_copy(ynn_subgraph_t subgraph, size_t num_stencils,
                                   const int32_t* stencil_axes,
                                   const int32_t* new_axes,
                                   const size_t* stencil_dims,
                                   const size_t* stencil_strides,
                                   const size_t* stencil_dilations,
                                   uint32_t input_id, uint32_t padding_id,
                                   uint32_t* output_id, uint32_t flags) {
  // Validate arguments.
  assert(subgraph);
  assert(subgraph->is_valid_value(input_id));
  assert(num_stencils == 0 || stencil_axes);
  assert(num_stencils == 0 || stencil_dims);
  assert(num_stencils == 0 || stencil_strides);
  assert(num_stencils == 0 || stencil_dilations);
  const ynn_value& input = subgraph->value(input_id);

  ynn_node::stencil_copy op;
  op.stencils.reserve(num_stencils);
  for (size_t i = 0; i < num_stencils; ++i) {
    assert(stencil_axes[i] >= 0);
    op.stencils.push_back(
        {axis_to_slinky_dim(input.rank(), stencil_axes[i]),
         axis_to_slinky_dim(input.rank() + num_stencils, new_axes[i]),
         stencil_dims[i], stencil_strides[i], stencil_dilations[i]});
  }
  // Sort the stencils so we can insert and remove the new axes while
  // understanding the axis indices.
  std::sort(op.stencils.begin(), op.stencils.end(),
            [](const stencil_info& a, const stencil_info& b) {
              return a.new_axis < b.new_axis;
            });

  const bool same_padding = padding_id != YNN_INVALID_VALUE_ID;

  // Propagate shape.
  ynn_value& output = subgraph->get_output_value(output_id, input);
  output.extents = input.extents;

  // Reduce the extents to avoid going out of bounds.
  for (const stencil_info& stencil : op.stencils) {
    slinky::index_t padding =
        same_padding ? 0 : dilated_kernel_size(stencil) - 1;
    output.extents[stencil.axis] =
        slinky::simplify(slinky::ceil_div<slinky::expr>(
            input.extents[stencil.axis] - padding, stencil.stride));
  }

  // Insert the new stencil dimensions.
  for (const stencil_info& stencil : op.stencils) {
    output.extents.insert(output.extents.begin() + stencil.new_axis,
                          static_cast<slinky::index_t>(stencil.extent));
  }

  // Make the node.
  ynn_node node;
  node.inputs = {input_id, padding_id};
  node.outputs = {output.id};
  node.op = std::move(op);
  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const ynn_node::stencil_copy& op =
        std::get<ynn_node::stencil_copy>(node.op);
    const int padding_id = node.inputs[1];
    const ynn_runtime_value& input = runtime.value(node.inputs[0]);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    slinky::buffer_expr_ptr input_buffer = input.buffer;

    if (padding_id != YNN_INVALID_VALUE_ID) {
      // Implement padding as a separate operation (as opposed to adding padding
      // to the copy below). Slinky can optimize this better if it is two copies
      // instead of one (both copies have the possibility of aliasing, but with
      // different operations).
      const ynn_runtime_value& padding = runtime.value(padding_id);
      slinky::buffer_expr_ptr padded =
          make_buffer_expr(runtime.symbols, "padded", input_buffer->rank(),
                           input_buffer->elem_size());

      const int rank = input.rank();
      std::vector<slinky::var> dims = make_dims(rank, runtime.symbols);
      slinky::func::input func_input{
          input_buffer, make_elementwise_bounds(dims, input.extents)};
      slinky::func::input func_padding{
          padding.buffer, make_elementwise_bounds(dims, padding.extents)};
      func_input.input_crop.resize(rank);
      for (int d = 0; d < rank; ++d) {
        auto stencil = std::find_if(
            op.stencils.begin(), op.stencils.end(),
            [d](const stencil_info& stencil) { return stencil.axis == d; });
        if (stencil == op.stencils.end()) continue;
        slinky::expr pre_padding =
            compute_same_padding_min(*stencil, input.extents[d]);
        func_input.bounds[d] -= pre_padding;
        if (input.extents[d].defined()) {
          func_input.input_crop[d] = slinky::min_extent(0, input.extents[d]);
        }
        func_padding.bounds[d] -= pre_padding;
      }

      auto func = slinky::func::make_copy(std::move(func_input), {padded, dims},
                                          std::move(func_padding));
      runtime.funcs.push_back(std::move(func));

      input_buffer = padded;
    }

    output.make_buffer(runtime);

    std::vector<slinky::var> dims =
        make_dims(output.buffer->rank(), runtime.symbols);

    std::vector<slinky::var> input_dims = dims;
    for (auto i = op.stencils.rbegin(); i != op.stencils.rend(); ++i) {
      input_dims.erase(input_dims.begin() + i->new_axis);
    }

    slinky::box_expr bounds =
        make_elementwise_bounds(input_dims, input.extents);
    for (const stencil_info& stencil : op.stencils) {
      bounds[stencil.axis] = point(dims[stencil.new_axis] * stencil.dilation +
                                   input_dims[stencil.axis] * stencil.stride);
    }

    auto func = slinky::func::make_copy({input_buffer, std::move(bounds)},
                                        {output.buffer, std::move(dims)});
    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

}  // extern "C"

}  // namespace ynn
