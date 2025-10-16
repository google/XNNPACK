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

#include "ynnpack/base/base.h"
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
auto make_get_tensor_shape_impl(std::vector<slinky::expr> extents) {
  return [=](const slinky::call_stmt* call,
             slinky::eval_context& ctx) -> slinky::index_t {
    assert(call->outputs.size() == 1);
    slinky::buffer<T, 1> shape = *ctx.lookup_buffer(call->outputs[0]);

    if (shape.rank == 0) {
      assert(extents.size() == 1);
      shape.at() = evaluate(extents[0], ctx);
    } else if (shape.rank == 1) {
      for (slinky::index_t i = shape.dim(0).begin(); i != shape.dim(0).end();
           ++i) {
        shape.at(i) = evaluate(extents[i], ctx);
      }
    } else {
      YNN_UNREACHABLE;
    }
    return 0;
  };
}

}  // namespace

extern "C" {

ynn_status ynn_define_get_tensor_shape(ynn_subgraph_t subgraph, size_t num_axes,
                                       const int32_t* axes, ynn_type type,
                                       size_t rank, uint32_t value_id,
                                       uint32_t* output_id, uint32_t flags) {
  // Validate arguments.
  assert(subgraph);
  assert(subgraph->is_valid_value(value_id));
  assert(output_id);
  if (*output_id == YNN_INVALID_VALUE_ID) {
    *output_id = subgraph->new_internal_value(type).id;
  }
  const ynn_value& input = subgraph->value(value_id);

  ynn_node::get_tensor_shape op;
  op.axes.resize(num_axes);
  for (size_t i = 0; i < num_axes; ++i) {
    op.axes[i] = axis_to_slinky_dim(input.rank(), axes[i]);
  }
  op.reshape_1d = (flags & YNN_NODE_FLAG_RESHAPE_1D) != 0;

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

    std::vector<slinky::var> dims = make_dims(output.rank(), runtime.symbols);

    std::vector<slinky::expr> extents;
    if (op.reshape_1d) {
      extents = {slinky::index_t(1)};
      for (int32_t i : op.axes) {
        if (input.extents[i].defined()) {
          extents[0] *= input.extents[i];
        }
      }
      extents[0] = slinky::simplify(extents[0]);
    } else {
      extents.reserve(op.axes.size());
      for (int32_t i : op.axes) {
        extents.push_back(input.extents[i].defined() ? input.extents[i] : 1);
      }
    }

    slinky::call_stmt::callable impl;
    switch (output.type) {
      case ynn_type_int32:
        impl = make_get_tensor_shape_impl<int32_t>(extents);
        break;
      case ynn_type_fp32:
        impl = make_get_tensor_shape_impl<float>(extents);
        break;
      default:
        YNN_UNREACHABLE;
    }

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
