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

#include "ynnpack/base/log.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/builder/pipeline.h"
#include "slinky/builder/simplify.h"
#include "slinky/runtime/expr.h"

namespace ynn {

namespace {

using slice_info = ynn_node::static_slice::slice;

std::pair<slinky::expr, slinky::expr> calc_begin_end(const slice_info& slice,
                                                     slinky::expr extent) {
  slinky::expr begin_expr, end_expr;

  if (slice.begin < 0) {
    begin_expr = slice.begin + extent;
  } else {
    begin_expr = slice.begin;
  }

  if (slice.end <= 0) {
    end_expr = slice.end + extent;
  } else {
    end_expr = slice.end;
  }

  begin_expr = slinky::clamp(begin_expr, 0, extent);
  end_expr = slinky::clamp(end_expr, 0, extent);

  return {begin_expr, end_expr};
}

}  // namespace

void define_static_slice(ynn_subgraph& subgraph, ynn_node& node,
                         uint32_t input_id, uint32_t output_id,
                         std::vector<ynn_node::static_slice::slice> slices,
                         bool slice_dims) {
  const ynn_value& input = subgraph.value(input_id);
  ynn_value& output = subgraph.value(output_id);

  std::reverse(slices.begin(), slices.end());

  ynn_node::static_slice op;
  op.slice_dims = slice_dims;
  op.slices = std::move(slices);

  // Propagate shape.
  output.extents = input.extents;
  for (const slice_info& slice : op.slices) {
    if (op.slice_dims) {
      assert(slice.axis < output.extents.size());
      output.extents.erase(output.extents.begin() + slice.axis);
    } else {
      auto begin_end = calc_begin_end(slice, output.extents[slice.axis]);
      const slinky::expr& begin = begin_end.first;
      const slinky::expr& end = begin_end.second;
      output.extents[slice.axis] = end - begin;
      if (slinky::prove_true(output.extents[slice.axis] == 1)) {
        output.extents[slice.axis] = {};
      }
    }
  }

  node.inputs = {input_id};
  node.outputs = {output_id};
  node.op = std::move(op);
  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const ynn_node::static_slice& op =
        std::get<ynn_node::static_slice>(node.op);
    const ynn_runtime_value& input = runtime.value(node.inputs[0]);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    output.make_buffer(runtime, input.buffer->elem_size());

    const int rank = input.rank();
    std::vector<slinky::var> dims = runtime.globals.make_dims(rank);
    slinky::func::input func_input{
        input.buffer, make_elementwise_bounds(dims, input.extents)};
    if (!op.slice_dims) {
      func_input.output_crop.resize(rank);
    }
    for (const slice_info& slice : op.slices) {
      const int d = slice.axis;
      auto begin_end = calc_begin_end(slice, input.extents[d]);
      if (op.slice_dims) {
        dims.erase(dims.begin() + d);
        func_input.bounds[d] = slinky::point(begin_end.first);
      } else {
        const slinky::expr& begin = begin_end.first;
        func_input.bounds[d] = func_input.bounds[d] + begin;
        func_input.output_crop[d] = all_bounds(output.extents[d]);
      }
    }
    auto func =
        slinky::func::make_copy(std::move(func_input), {output.buffer, dims});

    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };
}

extern "C" {

ynn_status ynn_define_static_slice(ynn_subgraph_t subgraph, size_t num_axes,
                                   const int32_t* axes, const int64_t* begins,
                                   const int64_t* ends, const int64_t* strides,
                                   uint32_t input_id, uint32_t* output_id,
                                   uint32_t flags) {
  const bool slice_dims = flags & YNN_NODE_FLAG_SLICE_DIMS;

  // Validate arguments.
  YNN_RETURN_IF_ERROR(validate_subgraph("static_slice", subgraph));
  YNN_RETURN_IF_ERROR(
      validate_input_tensor("static_slice", subgraph, "input_id", input_id));
  if (!slice_dims && (ends == nullptr || strides == nullptr)) {
    YNN_LOG_ERROR()
        << "For node `static_slice`, ends and strides must be non-null when "
           "YNN_NODE_FLAG_SLICE_DIMS is not set";
    return ynn_status_invalid_parameter;
  }
  YNN_RETURN_IF_ERROR(
      validate_output_tensor("static_slice", subgraph, "output_id", output_id));
  const ynn_value& input = subgraph->value(input_id);

  std::vector<slice_info> slices;
  slices.reserve(num_axes);
  for (int d = 0; d < num_axes; ++d) {
    const int32_t dim = axis_to_slinky_dim(input.rank(), axes[d]);
    if (dim >= 0 && dim < input.rank()) {
      slices.push_back({
          dim,
          begins[d],
          slice_dims ? 0 : ends[d],
          slice_dims ? 0 : strides[d],
      });
    } else {
      // The implicit dimensions are broadcasts, slicing them is a no-op.
      // TODO(dsharlet): I'm not sure if we want this feature or not. It is very
      // slinky-like to allow this, but not very XNNPACK-like.
    }
  }

  std::sort(
      slices.begin(), slices.end(),
      [](const slice_info& a, const slice_info& b) { return a.axis < b.axis; });

  // Propagate rank.
  ynn_value& output = subgraph->get_output_value(output_id, input);
  ynn_node node;
  define_static_slice(*subgraph, node, input_id, output.id, std::move(slices),
                      slice_dims);
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

}  // extern "C"

}  // namespace ynn
