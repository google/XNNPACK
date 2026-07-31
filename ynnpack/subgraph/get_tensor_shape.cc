// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/get_tensor_shape.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "ynnpack/base/base.h"
#include "ynnpack/base/log.h"
#include "ynnpack/base/span.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/builder/pipeline.h"
#include "slinky/builder/simplify.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/evaluate.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"
namespace ynn {
namespace {

template <typename T>
void implement_get_tensor_shape_typed(ynn::span<const slinky::expr> extents,
                                      const slinky::buffer<T>& output,
                                      slinky::eval_context* ctx) {
  if (output.rank == 0) {
    assert(extents.size() == 1);
    slinky::index_t val =
        ctx ? slinky::evaluate(extents[0], *ctx) : slinky::evaluate(extents[0]);
    output.at() = static_cast<T>(val);
  } else if (output.rank == 1) {
    for (slinky::index_t i = output.dim(0).begin(); i != output.dim(0).end();
         ++i) {
      slinky::index_t val = ctx ? slinky::evaluate(extents[i], *ctx)
                                : slinky::evaluate(extents[i]);
      output.at(i) = static_cast<T>(val);
    }
  } else {
    YNN_UNREACHABLE;
  }
}

void implement_get_tensor_shape(ynn::span<const slinky::expr> extents,
                                ynn_type type, const slinky::raw_buffer& output,
                                slinky::eval_context* ctx) {
  if (type == ynn_type_int32) {
    implement_get_tensor_shape_typed<int32_t>(extents, output.cast<int32_t>(),
                                              ctx);
  } else if (type == ynn_type_fp32) {
    implement_get_tensor_shape_typed<float>(extents, output.cast<float>(), ctx);
  } else {
    YNN_UNREACHABLE;
  }
}

std::vector<slinky::expr> get_tensor_shape_extents(
    ynn::span<const slinky::expr> extents, ynn::span<const int32_t> axes,
    bool reshape_1d) {
  std::vector<slinky::expr> selected;
  selected.reserve(axes.size());
  for (int32_t i : axes) {
    slinky::expr ext =
        (i < extents.size() && extents[i].defined()) ? extents[i] : 1;
    selected.push_back(ext);
  }

  if (reshape_1d) {
    slinky::expr extent = slinky::index_t(1);
    for (const slinky::expr& ext : selected) {
      extent *= ext;
    }
    return {slinky::simplify(extent)};
  } else {
    return selected;
  }
}

}  // namespace

void implement_get_tensor_shape(ynn::span<const slinky::expr> extents,
                                ynn::span<const int32_t> axes, bool reshape_1d,
                                ynn_type type, const slinky::raw_buffer& output,
                                slinky::eval_context* ctx) {
  implement_get_tensor_shape(
      get_tensor_shape_extents(extents, axes, reshape_1d), type, output, ctx);
}

extern "C" {

ynn_status ynn_define_get_tensor_shape(ynn_subgraph_t subgraph, size_t num_axes,
                                       const int32_t* axes, ynn_type type,
                                       size_t rank, uint32_t value_id,
                                       uint32_t* output_id, uint32_t flags) {
  // Validate arguments.
  YNN_RETURN_IF_ERROR(validate_subgraph("get_tensor_shape", subgraph));
  YNN_RETURN_IF_ERROR(validate_input_tensor("get_tensor_shape", subgraph,
                                            "value_id", value_id));
  YNN_RETURN_IF_ERROR(validate_output_tensor("get_tensor_shape", subgraph,
                                             "output_id", output_id));
  if (num_axes > 0 && axes == nullptr) {
    YNN_LOG_ERROR() << "For node `get_tensor_shape`, axes must be non-null "
                       "when num_axes > 0";
    return ynn_status_invalid_parameter;
  }
  if (*output_id == YNN_INVALID_VALUE_ID) {
    *output_id = subgraph->new_internal_value(type).id;
  }
  const ynn_value& input = subgraph->value(value_id);

  ynn_node::get_tensor_shape op;
  op.reshape_1d = (flags & YNN_NODE_FLAG_RESHAPE_1D) != 0;
  op.unique_dims = (flags & YNN_NODE_FLAG_UNIQUE_DIMS) != 0;
  op.axes.reserve(num_axes);
  for (size_t i = 0; i < num_axes; ++i) {
    int32_t axis = axis_to_slinky_dim(input.rank(), axes[i]);
    if (!op.unique_dims ||
        std::find(op.axes.begin(), op.axes.end(), axis) == op.axes.end()) {
      op.axes.push_back(axis);
    }
  }

  // Propagate shape.
  ynn_value& output = subgraph->value(*output_id);
  if (rank == 0) {
    output.extents.clear();
  } else if (op.reshape_1d) {
    output.extents = {static_cast<slinky::index_t>(1)};
  } else {
    output.extents = {static_cast<slinky::index_t>(op.axes.size())};
  }

  // Make the node.
  ynn_node node;
  node.inputs = {value_id};
  node.outputs = {*output_id};
  node.op = std::move(op);
  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const ynn_node::get_tensor_shape& op =
        std::get<ynn_node::get_tensor_shape>(node.op);
    const ynn_runtime_value& input = runtime.value(node.inputs[0]);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    output.make_buffer(runtime);

    std::vector<slinky::var> dims = runtime.globals.make_dims(output.rank());

    std::vector<slinky::expr> extents =
        get_tensor_shape_extents(input.extents, op.axes, op.reshape_1d);

    slinky::call_stmt::callable impl =
        [extents = std::move(extents), type = output.type](
            const slinky::call_stmt* call,
            slinky::eval_context& ctx) -> slinky::index_t {
      assert(call->outputs.size() == 1);
      const slinky::raw_buffer& shape = *ctx.lookup_buffer(call->outputs[0]);
      implement_get_tensor_shape(extents, type, shape, &ctx);
      return 0;
    };

    slinky::call_stmt::attributes attrs;
    attrs.name = "get_tensor_shape";
    auto func = slinky::func(std::move(impl), {},
                             {{output.buffer, std::move(dims)}}, {}, attrs);
    runtime.funcs.push_back(std::move(func));

    auto sched = std::make_unique<scheduling_info>();
    sched->force_root = true;

    runtime.funcs.back().user_data() = sched.get();
    runtime.scheduling_info_storage.push_back(std::move(sched));

    return ynn_status_success;
  };
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

}  // extern "C"

}  // namespace ynn
