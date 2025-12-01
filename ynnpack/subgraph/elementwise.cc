// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "ynnpack/base/log.h"
#include "ynnpack/base/to_string.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/binary/binary.h"
#include "ynnpack/kernels/ternary/ternary.h"
#include "ynnpack/kernels/unary/unary.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/utils.h"
#include "slinky/builder/pipeline.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

using ynn::operator<<;  // NOLINT(misc-unused-using-decls)

namespace ynn {

namespace {

inline const slinky::dim& dim_or_broadcast(const slinky::raw_buffer& buf,
                                           std::ptrdiff_t d) {
  return d < static_cast<std::ptrdiff_t>(buf.rank) ? buf.dim(d)
                                                   : slinky::dim::broadcast();
}

inline bool same_bounds(const slinky::dim& a, const slinky::dim& b) {
  // Return true if the dimensions have the same min and max or if they are
  // both broadcasts.
  return (a.min() == b.min() && a.max() == b.max()) ||
         (a.stride() == 0 && b.stride() == 0);
}

template <typename... Dims>
inline bool same_bounds(const slinky::dim& a, const slinky::dim& b,
                        const Dims&... dims) {
  return same_bounds(a, b) && same_bounds(b, dims...);
}

// A dimension is contiguous if it satisfies one of the following:
//   1. Its extent is 1. In this case, we disregard stride.
//   2. Its stride is equal to its element size.
inline bool is_continguous(const slinky::dim& dim, const int element_size) {
  return dim.extent() == 1 || dim.stride() == element_size;
}

// Call a unary kernel with a params buffer.
auto make_unary_elementwise_params_impl(unary_kernel_fn kernel) {
  return [kernel](slinky::raw_buffer a, slinky::raw_buffer params,
                  slinky::raw_buffer x) -> slinky::index_t {
    slinky::dim a_dims[2], x_dims[2];

    bool broadcast_params = true;
    const slinky::dim broadcast(0, 0, 0, 0);
    for (int i = 0; i < 2; ++i) {
      // Here, we can *only* support broadcasting of params in the kernel, so we
      // can only slice broadcasts. If we don't slice a dimension, use the
      // broadcast dimension instead.
      auto params_dim = dim_or_broadcast(params, 0);
      // TODO(dsharlet): Currently we only allow slicing m if we can slice n
      // first, which is a weird limitation.
      broadcast_params &= params_dim.stride() == 0;
      if (broadcast_params) {
        a_dims[i] = dim_or_broadcast(a, 0);
        x_dims[i] = dim_or_broadcast(x, 0);

        // `x` is already a view to the correct tile in the larger output
        // buffer. Inputs `a` and `params` are not. We must explicitly set their
        // offsets according to `x` before slicing.
        if (a.rank > 0) a.slice(0, x_dims[i].min());
        if (params.rank > 0) params.slice(0, x_dims[i].min());
        if (x.rank > 0) x.slice(0);
      } else {
        a_dims[i] = broadcast;
        x_dims[i] = broadcast;
      }

      while (x.rank > 0 && same_bounds(x_dims[i], a_dims[i]) &&
             same_bounds(x_dims[i], params_dim) &&
             slinky::can_fuse(x_dims[i], x.dim(0)) &&
             slinky::can_fuse(a_dims[i], dim_or_broadcast(a, 0)) &&
             slinky::can_fuse(params_dim, dim_or_broadcast(params, 0))) {
        params_dim = slinky::fuse(params_dim, dim_or_broadcast(params, 0));
        broadcast_params &= params_dim.stride() == 0;
        a_dims[i] = slinky::fuse(a_dims[i], dim_or_broadcast(a, 0));
        x_dims[i] = slinky::fuse(x_dims[i], x.dim(0));

        if (a.rank > 0) a.slice(0);
        if (params.rank > 0) params.slice(0);
        x.slice(0);
      }
      // Expect params dimension to be a broadcast.
      assert(!broadcast_params || params_dim.stride() == 0);
    }

    // We don't support broadcasting of `a` here in the innermost dimension (and
    // it would waste computation).
    assert(is_continguous(a_dims[0], a.elem_size));

    const slinky::dim& x_n = x_dims[0];
    const slinky::dim& a_m = a_dims[1];
    const slinky::dim& x_m = x_dims[1];

    slinky::for_each_element(
        [&](void* x, const void* a, const void* params) {
          kernel(x_m.extent(), x_n.extent(), a_m.stride(), a, x_m.stride(), x,
                 static_cast<const unary_params*>(params));
        },
        x, a, params);
    return 0;
  };
}

// Call a unary kernel without params.
auto make_unary_elementwise_impl(unary_kernel_fn kernel) {
  return
      [kernel](slinky::raw_buffer a, slinky::raw_buffer x) -> slinky::index_t {
        slinky::dim a_dims[2], x_dims[2];

        for (int i = 0; i < 2; ++i) {
          a_dims[i] = dim_or_broadcast(a, 0);
          x_dims[i] = dim_or_broadcast(x, 0);

          // `x` is already a view to the correct tile in the larger output
          // buffer. Inputs `a` and `b` are not. We must explicitly set their
          // offsets according to `x` before slicing.
          if (a.rank > 0) a.slice(0, x_dims[i].min());
          if (x.rank > 0) x.slice(0);

          while (x.rank > 0 && same_bounds(x_dims[i], a_dims[i]) &&
                 slinky::can_fuse(x_dims[i], x.dim(0)) &&
                 slinky::can_fuse(a_dims[i], dim_or_broadcast(a, 0))) {
            a_dims[i] = slinky::fuse(a_dims[i], dim_or_broadcast(a, 0));
            x_dims[i] = slinky::fuse(x_dims[i], x.dim(0));

            if (a.rank > 0) a.slice(0);
            x.slice(0);
          }
        }

        // We don't support broadcasting of `a` here in the innermost
        // dimension (and it would waste computation).
        assert(is_continguous(a_dims[0], a.elem_size));

        const slinky::dim& x_n = x_dims[0];
        const slinky::dim& a_m = a_dims[1];
        const slinky::dim& x_m = x_dims[1];

        slinky::for_each_element(
            [&](void* x, const void* a) {
              kernel(x_m.extent(), x_n.extent(), a_m.stride(), a, x_m.stride(),
                     x, nullptr);
            },
            x, a);
        return 0;
      };
}

// Binary kernels only support a single global params object, i.e. it must be
// globally broadcasted. Currently, the only operation that needs to support
// non-scalar params is `convert` with non-scalar quantization data.
// If we ever wanted to support binary operators with non-scalar quantization
// data, this would need to change.
auto make_binary_elementwise_impl(binary_kernel_fn kernel,
                                  const binary_params& params) {
  return [kernel, params](slinky::raw_buffer a, slinky::raw_buffer b,
                          slinky::raw_buffer x) -> slinky::index_t {
    slinky::dim a_dims[2], b_dims[2], x_dims[2];

    for (int i = 0; i < 2; ++i) {
      // If the output innermost (n) dimension has extent 1, we need to make the
      // n dimension of all inputs a broadcast. This case is not expected to
      // happen. For now, we add an assert to catch this case if it does.
      assert(i != 0 || is_continguous(dim_or_broadcast(x, 0), x.elem_size));

      a_dims[i] = dim_or_broadcast(a, 0);
      b_dims[i] = dim_or_broadcast(b, 0);
      x_dims[i] = dim_or_broadcast(x, 0);

      // `x` is already a view to the correct tile in the larger output buffer.
      // Inputs `a` and `b` are not. We must explicitly set their offsets
      // according to `x` before slicing.
      if (a.rank > 0) a.slice(0, x_dims[i].min());
      if (b.rank > 0) b.slice(0, x_dims[i].min());
      if (x.rank > 0) x.slice(0);

      while (x.rank > 0 && same_bounds(x_dims[i], a_dims[i], b_dims[i]) &&
             slinky::can_fuse(x_dims[i], x.dim(0)) &&
             slinky::can_fuse(a_dims[i], dim_or_broadcast(a, 0)) &&
             slinky::can_fuse(b_dims[i], dim_or_broadcast(b, 0))) {
        a_dims[i] = slinky::fuse(a_dims[i], dim_or_broadcast(a, 0));
        b_dims[i] = slinky::fuse(b_dims[i], dim_or_broadcast(b, 0));
        x_dims[i] = slinky::fuse(x_dims[i], x.dim(0));

        if (a.rank > 0) a.slice(0);
        if (b.rank > 0) b.slice(0);
        x.slice(0);
      }
    }

    const slinky::dim& a_n = a_dims[0];
    const slinky::dim& b_n = b_dims[0];
    const slinky::dim& x_n = x_dims[0];
    const slinky::dim& a_m = a_dims[1];
    const slinky::dim& b_m = b_dims[1];
    const slinky::dim& x_m = x_dims[1];

    slinky::for_each_element(
        [&](void* x, const void* a, const void* b) {
          kernel(x_m.extent(), x_n.extent(), a_m.stride(), a_n.stride(), a,
                 b_m.stride(), b_n.stride(), b, x_m.stride(), x, &params);
        },
        x, a, b);
    return 0;
  };
}

int compute_allow_in_place(const ynn_node& node, const ynn_subgraph& subgraph) {
  assert(node.outputs.size() == 1);
  int result = 0;
  for (int i = 0; i < node.inputs.size(); ++i) {
    if (allow_in_place(node.inputs[i], node.outputs[0], subgraph)) {
      result |= 1 << i;
    }
  }
  return result;
}

auto make_ternary_elementwise_impl(ternary_kernel_fn kernel) {
  return
      [kernel](slinky::raw_buffer a, slinky::raw_buffer b, slinky::raw_buffer c,
               slinky::raw_buffer x) -> slinky::index_t {
        slinky::dim a_dims[2], b_dims[2], c_dims[2], x_dims[2];

        for (int i = 0; i < 2; ++i) {
          // If the output innermost (n) dimension has extent 1, we need to make
          // the n dimension of all inputs a broadcast. This case is not
          // expected to happen. For now, we add an assert to catch this case if
          // it does.
          assert(i != 0 || is_continguous(dim_or_broadcast(x, 0), x.elem_size));

          a_dims[i] = dim_or_broadcast(a, 0);
          b_dims[i] = dim_or_broadcast(b, 0);
          c_dims[i] = dim_or_broadcast(c, 0);
          x_dims[i] = dim_or_broadcast(x, 0);

          // `x` is already a view to the correct tile in the larger output
          // buffer. Inputs `a` and `b` are not. We must explicitly set their
          // offsets according to `x` before slicing.
          if (a.rank > 0) a.slice(0, x_dims[i].min());
          if (b.rank > 0) b.slice(0, x_dims[i].min());
          if (c.rank > 0) c.slice(0, x_dims[i].min());
          if (x.rank > 0) x.slice(0);

          while (x.rank > 0 &&
                 same_bounds(x_dims[i], a_dims[i], b_dims[i], c_dims[i]) &&
                 slinky::can_fuse(x_dims[i], x.dim(0)) &&
                 slinky::can_fuse(a_dims[i], dim_or_broadcast(a, 0)) &&
                 slinky::can_fuse(b_dims[i], dim_or_broadcast(b, 0)) &&
                 slinky::can_fuse(c_dims[i], dim_or_broadcast(c, 0))) {
            a_dims[i] = slinky::fuse(a_dims[i], dim_or_broadcast(a, 0));
            b_dims[i] = slinky::fuse(b_dims[i], dim_or_broadcast(b, 0));
            c_dims[i] = slinky::fuse(c_dims[i], dim_or_broadcast(c, 0));
            x_dims[i] = slinky::fuse(x_dims[i], x.dim(0));

            if (a.rank > 0) a.slice(0);
            if (b.rank > 0) b.slice(0);
            if (c.rank > 0) c.slice(0);
            x.slice(0);
          }
        }

        const slinky::dim& a_n = a_dims[0];
        const slinky::dim& b_n = b_dims[0];
        const slinky::dim& c_n = c_dims[0];
        const slinky::dim& x_n = x_dims[0];
        const slinky::dim& a_m = a_dims[1];
        const slinky::dim& b_m = b_dims[1];
        const slinky::dim& c_m = c_dims[1];
        const slinky::dim& x_m = x_dims[1];

        slinky::for_each_element(
            [&](void* x, const void* a, const void* b, const void* c) {
              kernel(x_m.extent(), x_n.extent(), a_m.stride(), a_n.stride(), a,
                     b_m.stride(), b_n.stride(), b, c_m.stride(), c_n.stride(),
                     c, x_m.stride(), x, nullptr);
            },
            x, a, b, c);
        return 0;
      };
}

std::pair<float, int32_t> GetScalarQuantization(
    const ynn_runtime& runtime, const ynn_runtime_value& value) {
  std::pair<float, int32_t> result;
  result.first =
      value.scale_id != YNN_INVALID_VALUE_ID
          ? runtime.value(value.scale_id).static_scalar_value<float>()
          : 1.0f;
  result.second =
      value.zero_point_id != YNN_INVALID_VALUE_ID
          ? runtime.value(value.zero_point_id).static_scalar_value<int32_t>()
          : 0;
  return result;
}

ynn_status create_unary(const ynn_node& node, ynn_runtime& runtime,
                        unary_kernel_fn kernel) {
  assert(node.inputs.size() == 2);
  assert(node.outputs.size() == 1);

  const ynn_runtime_value& a = runtime.value(node.inputs[0]);
  ynn_runtime_value& x = runtime.value(node.outputs[0]);
  x.make_buffer(runtime);
  std::vector<slinky::var> dims = make_dims(x.rank(), runtime.symbols);
  slinky::box_expr bounds = make_elementwise_bounds(dims, a.extents);

  slinky::call_stmt::attributes attrs;
  attrs.name = to_string(std::get<ynn_node::unary_elementwise>(node.op).op);
  attrs.allow_in_place = compute_allow_in_place(node, runtime.subgraph);

  slinky::func func;
  if (node.inputs[1] != YNN_INVALID_VALUE_ID) {
    // This unary op has params in the second input argument.
    const ynn_runtime_value& params = runtime.value(node.inputs[1]);
    slinky::box_expr params_bounds =
        make_elementwise_bounds(dims, params.extents);

    func = slinky::func::make(make_unary_elementwise_params_impl(kernel),
                              {{a.buffer, std::move(bounds)},
                               {params.buffer, std::move(params_bounds)}},
                              {{x.buffer, dims}}, std::move(attrs));
  } else {
    func = slinky::func::make(make_unary_elementwise_impl(kernel),
                              {{a.buffer, std::move(bounds)}},
                              {{x.buffer, dims}}, std::move(attrs));
  }

  auto sched = runtime.make_schedule(dims, x.buffer, node.outputs[0]);
  func.user_data() = sched.get();
  runtime.scheduling_info_storage.push_back(std::move(sched));

  runtime.funcs.push_back(std::move(func));
  return ynn_status_success;
}

ynn_status create_binary(const ynn_node& node, ynn_runtime& runtime,
                         binary_kernel_fn kernel,
                         init_binary_params_fn init_params) {
  assert(node.inputs.size() == 2);
  assert(node.outputs.size() == 1);

  const ynn_runtime_value& a = runtime.value(node.inputs[0]);
  const ynn_runtime_value& b = runtime.value(node.inputs[1]);
  ynn_runtime_value& x = runtime.value(node.outputs[0]);

  x.make_buffer(runtime);

  binary_params params;
  if (init_params) {
    float a_scale;
    int32_t a_zero_point;
    float b_scale;
    int32_t b_zero_point;
    float x_scale;
    int32_t x_zero_point;
    std::tie(a_scale, a_zero_point) = GetScalarQuantization(runtime, a);
    std::tie(b_scale, b_zero_point) = GetScalarQuantization(runtime, b);
    std::tie(x_scale, x_zero_point) = GetScalarQuantization(runtime, x);
    init_params(a_scale, a_zero_point, b_scale, b_zero_point, x_scale,
                x_zero_point, params);
  }

  slinky::call_stmt::attributes attrs;
  attrs.name = to_string(std::get<ynn_node::binary_elementwise>(node.op).op);
  attrs.allow_in_place = compute_allow_in_place(node, runtime.subgraph);

  // Make the dims and bounds for this operation (does not depend on the
  // specific operation.)
  std::vector<slinky::var> dims = make_dims(x.rank(), runtime.symbols);
  slinky::box_expr a_bounds = make_elementwise_bounds(dims, a.extents);
  slinky::box_expr b_bounds = make_elementwise_bounds(dims, b.extents);
  a_bounds.resize(a.rank());
  b_bounds.resize(b.rank());
  auto func = slinky::func::make(
      make_binary_elementwise_impl(kernel, params),
      {{a.buffer, std::move(a_bounds)}, {b.buffer, std::move(b_bounds)}},
      {{x.buffer, dims}}, std::move(attrs));

  auto sched = runtime.make_schedule(dims, x.buffer, node.outputs[0]);
  func.user_data() = sched.get();
  runtime.scheduling_info_storage.push_back(std::move(sched));
  runtime.funcs.push_back(std::move(func));

  return ynn_status_success;
}

ynn_status create_ternary(const ynn_node& node, ynn_runtime& runtime,
                          ternary_kernel_fn kernel) {
  assert(node.inputs.size() == 3);
  assert(node.outputs.size() == 1);

  const ynn_runtime_value& a = runtime.value(node.inputs[0]);
  const ynn_runtime_value& b = runtime.value(node.inputs[1]);
  const ynn_runtime_value& c = runtime.value(node.inputs[2]);
  ynn_runtime_value& x = runtime.value(node.outputs[0]);

  x.make_buffer(runtime);

  slinky::call_stmt::attributes attrs;
  attrs.name = "ternary_elementwise";
  attrs.allow_in_place = compute_allow_in_place(node, runtime.subgraph);

  // Make the dims and bounds for this operation (does not depend on the
  // specific operation.)
  std::vector<slinky::var> dims = make_dims(x.rank(), runtime.symbols);
  slinky::box_expr a_bounds = make_elementwise_bounds(dims, a.extents);
  slinky::box_expr b_bounds = make_elementwise_bounds(dims, b.extents);
  slinky::box_expr c_bounds = make_elementwise_bounds(dims, c.extents);
  a_bounds.resize(a.rank());
  b_bounds.resize(b.rank());
  c_bounds.resize(c.rank());
  auto func = slinky::func::make(make_ternary_elementwise_impl(kernel),
                                 {{a.buffer, std::move(a_bounds)},
                                  {b.buffer, std::move(b_bounds)},
                                  {c.buffer, std::move(c_bounds)}},
                                 {{x.buffer, dims}}, attrs);

  auto sched = runtime.make_schedule(dims, x.buffer, node.outputs[0]);
  func.user_data() = sched.get();
  runtime.scheduling_info_storage.push_back(std::move(sched));
  runtime.funcs.push_back(std::move(func));

  return ynn_status_success;
}

void infer_shape(ynn_node& node, ynn_subgraph& subgraph) {
  ynn_value& x = subgraph.value(node.outputs[0]);
  int output_rank = 0;
  for (uint32_t i = 0; i < node.inputs.size(); ++i) {
    if (node.inputs[i] == YNN_INVALID_VALUE_ID) continue;
    output_rank = std::max(output_rank, subgraph.value(node.inputs[i]).rank());
  }
  x.extents.resize(output_rank);
  for (uint32_t i = 0; i < node.inputs.size(); ++i) {
    if (node.inputs[i] == YNN_INVALID_VALUE_ID) continue;
    for (size_t d = 0; d < x.rank(); ++d) {
      subgraph.infer_elementwise_shape(node, i, 0, d, d);
    }
  }
}

// Call the kernel's `init_params` function for each element of the input
// and output quantization data.
auto make_params_impl(init_unary_params_fn init_params) {
  return [init_params](
             slinky::buffer<const float, YNN_MAX_TENSOR_RANK> a_scale,
             slinky::buffer<const int32_t, YNN_MAX_TENSOR_RANK> a_zero_point,
             slinky::buffer<const float, YNN_MAX_TENSOR_RANK> x_scale,
             slinky::buffer<const int32_t, YNN_MAX_TENSOR_RANK> x_zero_point,
             const slinky::buffer<unary_params>& params) -> slinky::index_t {
    slinky::for_each_element(
        [&](unary_params* params, const float* a_scale,
            const int32_t* a_zero_point, const float* x_scale,
            const int32_t* x_zero_point) {
          init_params(*a_scale, *a_zero_point, *x_scale, *x_zero_point,
                      *params);
        },
        params, a_scale, a_zero_point, x_scale, x_zero_point);
    return 0;
  };
}

const ynn_runtime_value& value_or(const ynn_runtime& runtime, uint32_t id,
                                  const ynn_runtime_value& default_value) {
  if (id == YNN_INVALID_VALUE_ID) return default_value;
  return runtime.value(id);
}

// Make a node in the graph that calls the kernel's `init_params` function for
// each element of the input and output quantization data.
ynn_status define_make_unary_params(ynn_subgraph_t subgraph,
                                    init_unary_params_fn init_params,
                                    uint32_t input_a_id, uint32_t output_id,
                                    uint32_t params_id) {
  // Validate arguments.
  assert(subgraph);
  assert(subgraph->is_valid_value(input_a_id));
  assert(subgraph->is_valid_value(output_id));
  const ynn_value& a = subgraph->value(input_a_id);
  const ynn_value& x = subgraph->value(output_id);
  ynn_value& params = subgraph->value(params_id);

  ynn_node node;
  node.inputs = {a.scale_id, a.zero_point_id, x.scale_id, x.zero_point_id};
  node.outputs = {params_id};
  node.op = ynn_node::opaque{"make_unary_params"};

  // Propagate shape.
  params.extents.clear();
  params.extents.resize(params.rank());
  infer_shape(node, *subgraph);

  node.create = [init_params](const ynn_node& node, ynn_runtime& runtime) {
    ynn_runtime_value& params = runtime.value(node.outputs[0]);
    params.make_buffer(runtime, sizeof(unary_params));

    ynn_runtime_value one;
    one.buffer =
        slinky::buffer_expr::make_scalar<float>(runtime.symbols, "one", 1.0f);
    ynn_runtime_value zero;
    zero.buffer =
        slinky::buffer_expr::make_scalar<int32_t>(runtime.symbols, "zero", 0);
    const auto& a_scale = value_or(runtime, node.inputs[0], one);
    const auto& a_zero_point = value_or(runtime, node.inputs[1], zero);
    const auto& x_scale = value_or(runtime, node.inputs[2], one);
    const auto& x_zero_point = value_or(runtime, node.inputs[3], zero);

    // This is elementwise, but we allow implicit broadcasting from lower rank.
    std::vector<slinky::var> dims = make_dims(params.rank(), runtime.symbols);
    slinky::box_expr a_scale_bounds =
        make_elementwise_bounds(dims, a_scale.extents);
    slinky::box_expr a_zero_point_bounds =
        make_elementwise_bounds(dims, a_zero_point.extents);
    slinky::box_expr x_scale_bounds =
        make_elementwise_bounds(dims, x_scale.extents);
    slinky::box_expr x_zero_point_bounds =
        make_elementwise_bounds(dims, x_zero_point.extents);

    slinky::call_stmt::attributes attrs;
    attrs.name = "make_unary_params";

    auto func = slinky::func::make(
        make_params_impl(init_params),
        {
            {a_scale.buffer, std::move(a_scale_bounds)},
            {a_zero_point.buffer, std::move(a_zero_point_bounds)},
            {x_scale.buffer, std::move(x_scale_bounds)},
            {x_zero_point.buffer, std::move(x_zero_point_bounds)},
        },
        {{params.buffer, dims}}, std::move(attrs));

    auto sched = runtime.make_schedule(dims, params.buffer, node.outputs[0]);
    func.user_data() = sched.get();
    runtime.scheduling_info_storage.push_back(std::move(sched));

    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

}  // namespace

void define_binary(ynn_subgraph& subgraph, ynn_node& node, uint32_t input_a_id,
                   uint32_t input_b_id, uint32_t output_id,
                   ynn_binary_operator op, binary_kernel_fn kernel,
                   init_binary_params_fn init_params) {
  node.inputs = {input_a_id, input_b_id};
  node.outputs = {output_id};
  node.op = ynn_node::binary_elementwise{op};
  infer_shape(node, subgraph);
  node.create = [kernel, init_params](const ynn_node& node,
                                      ynn_runtime& runtime) {
    return create_binary(node, runtime, kernel, init_params);
  };
}

void define_ternary(ynn_subgraph& subgraph, ynn_node& node, uint32_t input_a_id,
                    uint32_t input_b_id, uint32_t input_c_id,
                    uint32_t output_id, ternary_kernel_fn kernel) {
  node.inputs = {input_a_id, input_b_id, input_c_id};
  node.outputs = {output_id};
  node.op = ynn_node::opaque{"ternary"};
  infer_shape(node, subgraph);
  node.create = [kernel](const ynn_node& node, ynn_runtime& runtime) {
    return create_ternary(node, runtime, kernel);
  };
}

extern "C" {

ynn_status ynn_define_unary(ynn_subgraph_t subgraph, ynn_unary_operator op,
                            uint32_t input_a_id, uint32_t* output_id,
                            uint32_t flags) {
  // Validate arguments.
  assert(subgraph);
  if (op == ynn_unary_invalid) {
    YNN_LOG_ERROR() << "invalid unary operator of input " << input_a_id;
    return ynn_status_invalid_parameter;
  }
  assert(subgraph->is_valid_value(input_a_id));
  assert(output_id);
  const ynn_value& a = subgraph->value(input_a_id);

  // Propagate rank.
  ynn_value& x = subgraph->get_output_value(output_id, a);
  x.extents.clear();
  x.extents.resize(x.rank());

  // Find the kernel.
  const bool a_is_quantized = a.scale_id != YNN_INVALID_VALUE_ID ||
                              a.zero_point_id != YNN_INVALID_VALUE_ID;
  const bool x_is_quantized = x.scale_id != YNN_INVALID_VALUE_ID ||
                              x.zero_point_id != YNN_INVALID_VALUE_ID;
  const unary_kernel* kernel =
      get_unary_kernel(op, a.type, a_is_quantized, x.type, x_is_quantized);
  if (!kernel) {
    YNN_LOG_ERROR() << "unsupported unary operator " << op << " for input type "
                    << a.type << " and output type " << x.type;
    return ynn_status_unsupported_parameter;
  }

  uint32_t params_id = YNN_INVALID_VALUE_ID;

  if (kernel->init_params) {
    // This kernel requires params. Define a node to construct the params.
    ynn_value& params = subgraph->new_internal_value();
    params.type = ynn_type_opaque;

    ynn_status status = define_make_unary_params(
        subgraph, kernel->init_params, input_a_id, *output_id, params.id);
    if (status != ynn_status_success) {
      return status;
    }

    if (params.rank() > 0) {
      // Params probably need broadcasting.
      int32_t axes[YNN_MAX_TENSOR_RANK];
      std::iota(axes, axes + params.rank(), 0);
      status = ynn_define_broadcast_like(subgraph, params.rank(),
                                         /*axes=*/axes, params.id, input_a_id,
                                         &params_id, /*flags=*/0);
      if (status != ynn_status_success) {
        return status;
      }
    } else {
      params_id = params.id;
    }
  }

  // Make the node.
  ynn_node node;
  node.inputs = {input_a_id, params_id};
  node.outputs = {*output_id};
  node.op = ynn_node::unary_elementwise{op};
  infer_shape(node, *subgraph);
  node.create = [kernel = kernel->op](const ynn_node& node,
                                      ynn_runtime& runtime) {
    return create_unary(node, runtime, kernel);
  };
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

ynn_status ynn_define_binary(ynn_subgraph_t subgraph, ynn_binary_operator op,
                             uint32_t input_a_id, uint32_t input_b_id,
                             uint32_t* output_id, uint32_t flags) {
  // Validate arguments.
  assert(subgraph);
  if (op == ynn_binary_invalid) {
    YNN_LOG_ERROR() << "invalid binary operator of inputs " << input_a_id
                    << ", " << input_b_id;
    return ynn_status_invalid_parameter;
  }
  assert(subgraph->is_valid_value(input_a_id));
  assert(subgraph->is_valid_value(input_b_id));
  assert(output_id);
  const ynn_value& a = subgraph->value(input_a_id);
  const ynn_value& b = subgraph->value(input_b_id);

  // Propagate rank.
  ynn_value& x = subgraph->get_output_value(output_id, a);
  assert(a.type == b.type);
  assert(a.type == x.type);

  // Find the kernel.
  const bool is_quantized = a.scale_id != YNN_INVALID_VALUE_ID ||
                            a.zero_point_id != YNN_INVALID_VALUE_ID ||
                            b.scale_id != YNN_INVALID_VALUE_ID ||
                            b.zero_point_id != YNN_INVALID_VALUE_ID ||
                            x.scale_id != YNN_INVALID_VALUE_ID ||
                            x.zero_point_id != YNN_INVALID_VALUE_ID;
  const binary_kernel* kernel = get_binary_kernel(op, x.type, is_quantized);
  if (!kernel) {
    YNN_LOG_ERROR() << "unsupported binary operator " << op
                    << " for input types " << a.type << ", " << b.type
                    << " and output type " << x.type;
    return ynn_status_unsupported_parameter;
  }

  ynn_node node;
  node.inputs = {input_a_id, input_b_id};
  node.outputs = {*output_id};
  node.op = ynn_node::binary_elementwise{op};
  infer_shape(node, *subgraph);
  node.create = [kernel](const ynn_node& node, ynn_runtime& runtime) {
    return create_binary(node, runtime, kernel->op, kernel->init_params);
  };
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

}  // extern "C"

}  // namespace ynn
