// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/reduce/reduce.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "ynnpack/base/base.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/utils.h"
#include "slinky/base/arithmetic.h"
#include "slinky/builder/pipeline.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

namespace ynn {

namespace {

unary_reduce_kernel_fn get_reduce_kernel(ynn_reduce_operator op,
                                         ynn_type a_type, ynn_type c_type) {
  switch (op) {
    case ynn_reduce_sum:
      return get_sum_kernel(a_type, c_type);
    case ynn_reduce_sum_squared:
      return get_sum_squared_kernel(a_type, c_type);
    case ynn_reduce_max:
      return get_max_kernel(a_type, c_type);
    case ynn_reduce_min:
      return get_min_kernel(a_type, c_type);
    case ynn_reduce_min_max:
      return get_min_max_kernel(a_type, c_type);
    default:
      return nullptr;
  }
}

// A reduction output is a buffer that has the same dimensions as the input,
// but with stride 0 for the reduction dimensions.
void prepare_reduction_output(slinky::buffer<void, YNN_MAX_TENSOR_RANK>& output,
                              const slinky::raw_buffer& input,
                              const ynn::axes_set& k_dims) {
  if (output.rank != input.rank) {
    for (int d = 0; d < input.rank; ++d) {
      if (k_dims[d]) {
        slinky::dim new_dim = input.dim(d);
        new_dim.set_stride(0);
        output.unslice(d, new_dim);
      }
    }
    assert(output.rank == input.rank);
  } else {
    for (int d = 0; d < input.rank; ++d) {
      if (k_dims[d]) {
        output.dim(d) = input.dim(d);
        output.dim(d).set_stride(0);
      }
    }
  }
}

// The wrapper for the kernel we use when we actually want to run a reduce
// kernel on some buffers.
auto make_unary_reduce_impl(ynn_reduce_operator op,
                            unary_reduce_kernel_fn kernel,
                            ynn::axes_set k_dims) {
  return [op, kernel, k_dims](
             slinky::buffer<const void, YNN_MAX_TENSOR_RANK> a,
             slinky::buffer<const void, YNN_MAX_TENSOR_RANK> init_c,
             slinky::buffer<void, YNN_MAX_TENSOR_RANK> c) -> slinky::index_t {
    if (init_c.base() == c.base()) {
      // The input and accumulator were aliased to the same buffer, we don't
      // need to copy it.
      // TODO: Do we need to slice init_c first? Or maybe just fall through to
      // slinky::copy and make it optimize this case?
    } else {
      allow_broadcasting(init_c);
      slinky::copy(init_c, c);
    }

    // Slice off the "channel" dimension if any.
    slinky::index_t c_stride_m = 0;
    if (op == ynn_reduce_min_max) {
      c_stride_m = c.dim(c.rank - 1).stride();
      c.rank -= 1;
    }

    // Make the output have the same rank as the iput, but with stride 0 dims
    // where we want to do a reduction.
    prepare_reduction_output(c, a, k_dims);

    // The next bit of logic selects which dimensions will be handled by the
    // kernel. We start out with the kernel's dimensions as no-ops (1). Any time
    // we want to send a dimension to the kernel, we just have to find one of
    // these that has extent 1, and then it can be replaced.
    size_t n = 1;
    size_t k1 = 1;
    size_t k2 = 1;
    size_t k3 = 1;
    size_t a_stride_n = 0;
    assert(a.dim(0).stride() == a.elem_size || a.dim(0).extent() == 1);
    if (k_dims[0]) {
      // The dense dimension is reduced.
      k1 = a.dim(0).extent();
    } else {
      n = c.dim(0).extent();
      a_stride_n = a.dim(0).stride();
    }

    // A helper to track slicing
    int sliced = 0;
    auto slice = [&](int d) {
      assert(!a.dim(d).is_folded(c.dim(d)));
      assert(!c.dim(d).is_folded());
      a.slice(d, c.dim(d).min());
      c.slice(d);
      ++sliced;
    };

    // We took dimension 0 above.
    slice(0);

    size_t a_stride_k3 = 0;
    size_t a_stride_k2 = 0;
    for (int i = 0; i < a.rank;) {
      if (k_dims[i + sliced]) {
        if (k2 == 1) {
          k2 = a.dim(i).extent();
          a_stride_k2 = a.dim(i).stride();
          slice(i);
          continue;
        } else if (k3 == 1) {
          k3 = a.dim(i).extent();
          a_stride_k3 = a.dim(i).stride();
          slice(i);
          continue;
        }
      } else {
        // Not a reduction dimension. If we haven't already found a dimension
        // for the kernel, give it this one.
        if (n == 1 && c.dim(i).stride() == c.elem_size) {
          n = c.dim(i).extent();
          a_stride_n = a.dim(i).stride();
          slice(i);
          continue;
        }
      }
      // If we get here, we are keeping this dimension.
      ++i;
    }

    slinky::for_each_element(
        [&](void* c, const void* a) {
          kernel(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2, a,
                 c_stride_m, c);
        },
        c, a);

    return 0;
  };
}

ynn_type get_accumulator_type(ynn_reduce_operator op, ynn_type a_type) {
  switch (op) {
    case ynn_reduce_sum:
    case ynn_reduce_sum_squared:
      return ynn::type_is_integral(a_type) ? ynn_type_int32 : ynn_type_fp32;
    case ynn_reduce_max:
    case ynn_reduce_min:
    case ynn_reduce_min_max:
      return a_type;
    default:
      YNN_UNREACHABLE;
  }
}

uint32_t get_reduce_identity_value(ynn_subgraph_t subgraph,
                                   const ynn_value& output,
                                   ynn_reduce_operator op) {
  float value_f32[2];
  int rank = 0;
  size_t dims[YNN_MAX_TENSOR_RANK];
  std::fill_n(dims, YNN_MAX_TENSOR_RANK, 1);
  switch (op) {
    case ynn_reduce_sum:
    case ynn_reduce_sum_squared:
      value_f32[0] = 0.0f;
      break;
    case ynn_reduce_max:
      value_f32[0] = -std::numeric_limits<float>::infinity();
      break;
    case ynn_reduce_min:
      value_f32[0] = std::numeric_limits<float>::infinity();
      break;
    case ynn_reduce_min_max:
      value_f32[0] = std::numeric_limits<float>::infinity();
      value_f32[1] = -std::numeric_limits<float>::infinity();
      rank = output.rank();
      dims[rank - 1] = 2;
      break;
    default:
      return YNN_INVALID_VALUE_ID;
  }

  uint32_t zero_point_id;
  uint32_t scale_id;
  switch (op) {
    case ynn_reduce_sum:
    case ynn_reduce_sum_squared:
      // Here, we want the unquantized identity value.
      // TODO(dsharlet): Why? I think it's because we are ignoring these
      // quantization parameters when implementing the sum reduction. This is a
      // bit of a wart in the design here.
      zero_point_id = YNN_INVALID_VALUE_ID;
      scale_id = YNN_INVALID_VALUE_ID;
      break;
    case ynn_reduce_max:
    case ynn_reduce_min:
    case ynn_reduce_min_max:
      zero_point_id = output.zero_point_id;
      scale_id = output.scale_id;
      break;
    default:
      return YNN_INVALID_VALUE_ID;
  }

  // TODO(dsharlet): `get_static_value_id` uses public API functions to create
  // the tensor, which expect the dimensions in descending stride order.
  std::reverse(dims, dims + rank);

  return subgraph->get_static_value_id(output.type, rank, dims, zero_point_id,
                                       scale_id, value_f32);
}

}  // namespace

extern "C" {

ynn_status ynn_define_reduce(ynn_subgraph_t subgraph,
                             enum ynn_reduce_operator op, size_t num_axes,
                             const int32_t* axes, uint32_t input_a_id,
                             uint32_t input_b_id, uint32_t* output_id,
                             uint32_t flags) {
  // Validate arguments.
  assert(subgraph);
  assert(subgraph->is_valid_value(input_a_id));
  const ynn_value& a = subgraph->value(input_a_id);

  ynn_node::reduce reduce;
  reduce.op = op;
  for (size_t i = 0; i < num_axes; ++i) {
    reduce.k_dims[axis_to_slinky_dim(a.rank(), axes[i])] = true;
  }
  reduce.keep_dims = flags & YNN_NODE_FLAG_KEEP_DIMS;

  assert(output_id);
  if (*output_id == YNN_INVALID_VALUE_ID) {
    // Make the output for this reduction.
    ynn_type output_type = get_accumulator_type(op, a.type);
    ynn_value& output = subgraph->new_internal_value(output_type);
    uint32_t reduce_size_id = YNN_INVALID_VALUE_ID;
    switch (op) {
      case ynn_reduce_sum:
      case ynn_reduce_sum_squared:
        if (a.zero_point_id != YNN_INVALID_VALUE_ID) {
          // When computing a sum, the zero point gets multiplied by the number
          // of elements in the reduction.
          ynn_define_get_tensor_shape(subgraph, num_axes, axes, ynn_type_int32,
                                      /*rank=*/0, input_a_id, &reduce_size_id,
                                      /*flags=*/YNN_NODE_FLAG_RESHAPE_1D);
          ynn_define_binary(subgraph, ynn_binary_multiply, a.zero_point_id,
                            reduce_size_id, &output.zero_point_id,
                            /*flags=*/0);
        }
        output.scale_id = a.scale_id;
        break;
      case ynn_reduce_max:
      case ynn_reduce_min:
      case ynn_reduce_min_max:
        output.zero_point_id = a.zero_point_id;
        output.scale_id = a.scale_id;
        break;
      default:
        YNN_UNREACHABLE;
    }

    *output_id = output.id;
  }

  // Propagate shape
  ynn_value& output = subgraph->value(*output_id);
  output.extents = a.extents;
  for (int i = static_cast<int>(output.extents.size()) - 1; i >= 0; --i) {
    if (reduce.k_dims[i]) {
      if (reduce.keep_dims) {
        output.extents[i] = {};
      } else {
        output.extents.erase(output.extents.begin() + i);
      }
    }
  }

  if (op == ynn_reduce_min_max) {
    // This reduction adds a dimension for the min/max index.
    output.extents.push_back(2);
  }

  if (input_b_id == YNN_INVALID_VALUE_ID) {
    input_b_id = get_reduce_identity_value(subgraph, output, op);
  }

  // Get the reduce kernel we are going to use.
  unary_reduce_kernel_fn kernel = get_reduce_kernel(op, a.type, output.type);
  assert(kernel);

  // Make the node.
  ynn_node node;
  node.inputs = {input_a_id, input_b_id};
  node.outputs = {*output_id};
  node.op = std::move(reduce);

  node.create = [kernel](const ynn_node& node, ynn_runtime& runtime) {
    const ynn_node::reduce& op = std::get<ynn_node::reduce>(node.op);
    const ynn_runtime_value& input_a = runtime.value(node.inputs[0]);
    const ynn_runtime_value& input_c = runtime.value(node.inputs[1]);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    output.make_buffer(runtime);

    std::vector<slinky::var> dims = make_dims(input_a.rank(), runtime.symbols);
    slinky::box_expr a_bounds = make_elementwise_bounds(dims, input_a.extents);
    slinky::box_expr c_bounds = make_elementwise_bounds(dims, input_c.extents);

    for (int i = static_cast<int>(input_a.rank()) - 1; i >= 0; --i) {
      if (op.k_dims[i]) {
        a_bounds[i] = all_bounds(input_a.extents[i]);
        if (!op.keep_dims) {
          c_bounds.erase(c_bounds.begin() + i);
          dims.erase(dims.begin() + i);
        }
      }
    }

    slinky::call_stmt::attributes attrs;
    attrs.name = node.to_string();
    // Allow the input_c and output to be computed in-place, which means we
    // don't need to initialize the accumulator.
    if (allow_in_place(input_c.id, output.id, runtime.subgraph)) {
      attrs.allow_in_place = (1 << 1);
    }
    auto sched = std::make_unique<scheduling_info>();
    if (!dims.empty()) {
      // The elementwise schedule is based on the output shape.
      // The cost of computation of a single output element is modeled
      // as the product of the reduction dimensions multiplied by the element
      // size and divided by some tuning coefficient. This naturally leads to
      // smaller tiles for large reductions and bigger tiles for small ones.
      static constexpr int cost_scaling_factor = 512;  // 256;
      slinky::expr reduction_cost = input_a.buffer->elem_size();
      for (int d = 0; d < input_a.rank(); ++d) {
        if (op.k_dims[d] && input_a.extents[d].defined()) {
          reduction_cost *= input_a.extents[d];
        }
      }
      reduction_cost =
          slinky::ceil_div(reduction_cost, slinky::expr(cost_scaling_factor));
      sched = runtime.make_schedule(dims, output.buffer, node.outputs[0], {},
                                    reduction_cost);
    } else {
      // This is a total reduction, so can't have any loops and we don't want
      // to schedule it inside of any other loops.
      sched->force_root = true;
    }
    auto func = slinky::func::make(
        make_unary_reduce_impl(op.op, kernel, op.k_dims),
        {{input_a.buffer, std::move(a_bounds)},
         {input_c.buffer, std::move(c_bounds)}},
        {{output.buffer, std::move(dims)}}, std::move(attrs));

    func.user_data() = sched.get();
    runtime.scheduling_info_storage.push_back(std::move(sched));
    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

}  // extern "C"

}  // namespace ynn
