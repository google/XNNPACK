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
#include "ynnpack/subgraph/reduce.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/utils.h"
#include "slinky/base/arithmetic.h"
#include "slinky/builder/pipeline.h"
#include "slinky/builder/simplify.h"
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

// The wrapper for the kernel we use when we actually want to run a reduce
// kernel on some buffers.
auto make_unary_reduce_impl(const ynn_node::reduce& op,
                            unary_reduce_kernel_fn kernel) {
  return [op, kernel](
             slinky::buffer<const void, YNN_MAX_TENSOR_RANK> a,
             slinky::buffer<const void, YNN_MAX_TENSOR_RANK> init_c,
             slinky::buffer<void, YNN_MAX_TENSOR_RANK> c) -> slinky::index_t {
    if (init_c.base() == c.base()) {
      // The input and accumulator were aliased to the same buffer, we don't
      // need to copy it.
      // TODO: Do we need to slice init_c first? Or maybe just fall through to
      // slinky::copy and make it optimize this case?
    } else {
      slinky::copy(init_c, c);
    }

    // Slice off the "channel" dimension if any.
    slinky::index_t c_stride_m = 0;
    if (op.op == ynn_reduce_min_max) {
      c_stride_m = c.dim(c.rank - 1).stride();
      c.rank -= 1;
    }

    // Conceptually, a reduction is c = f(c, a) for all indices of a. We need to
    // do a few things to implement this:
    // 1. We need broadcast dimensions in c corresponding to the reduction
    // dimensions in a. If `keep_dims` is false, that means inserting new
    // broadcast dimensions.
    // 2. Remove the dimensions handled by the kernel.
    // 3. We can optimize the loops over the buffers by fusing contiguous
    // dimensions where possible.
    size_t n = 1;
    size_t k1 = 1;
    size_t k2 = 1;
    size_t k3 = 1;
    size_t a_stride_n = 0;

    size_t a_stride_k3 = 0;
    size_t a_stride_k2 = 0;

    int sliced = 0;
    for (int i = 0; i < a.rank;) {
      const slinky::dim& a_dim_i = a.dim(i);
      slinky::index_t extent_i = a_dim_i.extent();
      if (extent_i == 0) {
        // The reduction is an empty reduction, so we are done after
        // initializing the output.
        return 0;
      } else if (op.k_dims[i + sliced]) {
        assert(a_dim_i.min() == 0);
        assert(!a_dim_i.is_folded());
        if (extent_i == 1) {
          // Slice extent 1 dimensions, they don't affect the result.
        } else if (k1 * a.elem_size == a_dim_i.stride()) {
          k1 *= extent_i;
        } else if (a_stride_k2 * k2 == a_dim_i.stride()) {
          k2 *= extent_i;
        } else if (a_stride_k3 * k3 == a_dim_i.stride()) {
          k3 *= extent_i;
        } else if (k2 == 1) {
          k2 = extent_i;
          a_stride_k2 = a_dim_i.stride();
        } else if (k3 == 1) {
          k3 = extent_i;
          a_stride_k3 = a_dim_i.stride();
        } else {
          // This is a reduction dimension that is not handled by the kernel.
          if (op.keep_dims) {
            // Replace the existing dimension in c with a broadcast.
            c.mutable_dim(i).set_bounds(a_dim_i.min(), a_dim_i.max());
            c.mutable_dim(i).set_stride(0);
          } else {
            // Add a new dimension for the broadcast in c.
            slinky::dim new_dim = a_dim_i;
            new_dim.set_stride(0);
            c.unslice(i, new_dim);
          }
          ++i;
          continue;
        }
        // This is a reduction dimension that is handled by the kernel.
        a.slice(i);
        if (op.keep_dims) {
          // If op.keep_dims is true, that means the buffer has these
          // dimensions, but we don't want them there for dimensions we handle
          // with the kernel.
          c.slice(i);
        }
        ++sliced;
      } else {
        const slinky::dim& c_dim_i = c.dim(i);
        // Not a reduction dimension. If we haven't already found a dimension
        // for the kernel, give it this one.
        if (extent_i == 1) {
          // Slice extent 1 dimensions, they don't affect the result.
        } else if (c_dim_i.stride() == n * c.elem_size) {
          if (a_stride_n * n == a_dim_i.stride()) {
            n *= c_dim_i.extent();
          } else if (n == 1) {
            n = c_dim_i.extent();
            a_stride_n = a_dim_i.stride();
          } else {
            ++i;
            continue;
          }
        } else {
          ++i;
          continue;
        }
        assert(!a_dim_i.is_folded(c_dim_i));
        assert(!c_dim_i.is_folded());
        a.slice(i, c_dim_i.min());
        c.slice(i);
        ++sliced;
      }
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
  if (a_type == ynn_type_fp64) {
    return ynn_type_fp64;
  }
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

uint32_t get_reduce_identity_value(ynn_subgraph& subgraph,
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

  return subgraph.get_static_value_id(output.type, rank, dims, zero_point_id,
                                      scale_id, value_f32);
}

}  // namespace

void define_reduce(ynn_subgraph& subgraph, ynn_node& node,
                   ynn_reduce_operator op, const ynn::axes_set& k_dims,
                   uint32_t input_a_id, uint32_t input_b_id,
                   uint32_t* output_id, bool keep_dims) {
  const ynn_value& a = subgraph.value(input_a_id);

  if (*output_id == YNN_INVALID_VALUE_ID) {
    // Make the output for this reduction.
    ynn_type output_type = get_accumulator_type(op, a.type);
    ynn_value& output = subgraph.new_internal_value(output_type);
    uint32_t reduce_size_id = YNN_INVALID_VALUE_ID;
    switch (op) {
      case ynn_reduce_sum:
      case ynn_reduce_sum_squared:
        if (a.zero_point_id != YNN_INVALID_VALUE_ID) {
          // When computing a sum, the zero point gets multiplied by the number
          // of elements in the reduction.
          int32_t axes[YNN_MAX_TENSOR_RANK];
          int num_axes = 0;
          for (int i = 0; i < a.rank(); ++i) {
            if (k_dims[i]) axes[num_axes++] = a.rank() - 1 - i;
          }
          ynn_define_get_tensor_shape(&subgraph, num_axes, axes, ynn_type_int32,
                                      /*rank=*/0, input_a_id, &reduce_size_id,
                                      /*flags=*/YNN_NODE_FLAG_RESHAPE_1D);
          ynn_define_binary(&subgraph, ynn_binary_multiply, a.zero_point_id,
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

  ynn_value& output = subgraph.value(*output_id);

  // Get the reduce kernel we are going to use.
  unary_reduce_kernel_fn kernel = get_reduce_kernel(op, a.type, output.type);
  assert(kernel);

  // Propagate shape
  output.extents = a.extents;
  for (int i = static_cast<int>(output.extents.size()) - 1; i >= 0; --i) {
    if (k_dims[i]) {
      if (keep_dims) {
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
  } else {
    const ynn_value& b = subgraph.value(input_b_id);
    if (b.type != output.type) {
      input_b_id = YNN_INVALID_VALUE_ID;
      ynn_define_convert(&subgraph, b.id, output.type, output.zero_point_id,
                         output.scale_id, &input_b_id, /*flags=*/0);
    }
  }

  node.inputs = {input_a_id, input_b_id};
  node.outputs = {*output_id};
  node.op = ynn_node::reduce{k_dims, op, keep_dims};

  node.create = [kernel](const ynn_node& node, ynn_runtime& runtime) {
    const ynn_node::reduce& op = std::get<ynn_node::reduce>(node.op);
    const ynn_runtime_value& input_a = runtime.value(node.inputs[0]);
    const ynn_runtime_value& input_c = runtime.value(node.inputs[1]);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    output.make_buffer(runtime);

    std::vector<slinky::var> dims = runtime.globals.make_dims(input_a.rank());
    slinky::box_expr a_bounds = make_elementwise_bounds(dims, input_a.extents);

    for (int i = static_cast<int>(input_a.rank()) - 1; i >= 0; --i) {
      if (op.k_dims[i]) {
        a_bounds[i] = all_bounds(input_a.extents[i]);
        if (!op.keep_dims) {
          dims.erase(dims.begin() + i);
        }
      }
    }

    slinky::box_expr c_bounds = make_elementwise_bounds(dims, input_c.extents);

    slinky::call_stmt::attributes attrs;
    attrs.name = node.to_string();
    // Allow the input_c and output to be computed in-place, which means we
    // don't need to initialize the accumulator.
    if (allow_in_place(input_c.id, output.id, *runtime.subgraph)) {
      attrs.allow_in_place = (1 << 1);
    }
    auto sched = std::make_unique<scheduling_info>();
    slinky::expr output_count = 1;
    for (const slinky::expr& e : output.extents) {
      if (e.defined()) {
        output_count *= e;
      }
    }
    if (dims.empty() || slinky::prove_true(output_count == 1)) {
      // This is a total reduction, so can't have any loops and we don't want
      // to schedule it inside of any other loops.
      sched->force_root = true;
    } else {
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
    }
    auto func = slinky::func::make(
        make_unary_reduce_impl(op, kernel),
        {{input_a.buffer, std::move(a_bounds)},
         {input_c.buffer, std::move(c_bounds)}},
        {{output.buffer, std::move(dims)}}, std::move(attrs));

    func.user_data() = sched.get();
    runtime.scheduling_info_storage.push_back(std::move(sched));
    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };
}

extern "C" {

ynn_status ynn_define_reduce(ynn_subgraph_t subgraph,
                             enum ynn_reduce_operator op, size_t num_axes,
                             const int32_t* axes, uint32_t input_a_id,
                             uint32_t input_b_id, uint32_t* output_id,
                             uint32_t flags) {
  // Validate arguments.
  YNN_RETURN_IF_ERROR(validate_subgraph("reduce", subgraph));
  YNN_RETURN_IF_ERROR(
      validate_input_tensor("reduce", subgraph, "input_a_id", input_a_id));
  YNN_RETURN_IF_ERROR(validate_input_tensor("reduce", subgraph, "input_b_id",
                                            input_b_id, /*optional=*/true));
  YNN_RETURN_IF_ERROR(
      validate_output_tensor("reduce", subgraph, "output_id", output_id));

  const ynn_value& a = subgraph->value(input_a_id);

  ynn::axes_set k_dims;
  for (size_t i = 0; i < num_axes; ++i) {
    const int axis = axis_to_slinky_dim(a.rank(), axes[i]);
    if (axis < a.rank()) {
      k_dims[axis] = true;
    } else {
      // This is a reduction of an implicit broadcast, which is a no-op.
    }
  }
  bool keep_dims = flags & YNN_NODE_FLAG_KEEP_DIMS;

  uint32_t convert_to_id = YNN_INVALID_VALUE_ID;
  if (*output_id != YNN_INVALID_VALUE_ID) {
    ynn_type output_type = get_accumulator_type(op, a.type);
    if (subgraph->value(*output_id).type != output_type) {
      // We need to compute the reduction into an intermediate accumulator, and
      // convert to the output after.
      convert_to_id = *output_id;
      *output_id = YNN_INVALID_VALUE_ID;
    }
  }

  // Make the node.
  ynn_node node;
  define_reduce(*subgraph, node, op, k_dims, input_a_id, input_b_id, output_id,
                keep_dims);
  subgraph->add_node(std::move(node));

  if (convert_to_id != YNN_INVALID_VALUE_ID) {
    YNN_RETURN_IF_ERROR(ynn_define_unary(subgraph, ynn_unary_convert,
                                         *output_id, &convert_to_id, flags));
    *output_id = convert_to_id;
  }

  return ynn_status_success;
}

}  // extern "C"

}  // namespace ynn
