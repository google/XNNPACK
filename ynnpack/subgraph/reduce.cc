// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/reduce/reduce.h"

#include <algorithm>
#include <cassert>
#include <cmath>
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
#include "ynnpack/subgraph/copy.h"
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

float get_reduce_identity(ynn_reduce_operator op) {
  switch (op) {
    case ynn_reduce_sum:
    case ynn_reduce_sum_squared:
      return 0.0f;
    case ynn_reduce_max:
      return -std::numeric_limits<float>::infinity();
    case ynn_reduce_min:
      return std::numeric_limits<float>::infinity();
    default:
      return std::nan("");
  }
}

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
             slinky::buffer<void, YNN_MAX_TENSOR_RANK> c,
             const slinky::raw_buffer& reduction_bounds) -> slinky::index_t {
    // Determine if this is the first tile in the reduction.
    bool init_output = true;
    for (int i = 0; i < reduction_bounds.rank; ++i) {
      if (reduction_bounds.dim(i).min() != 0) {
        init_output = false;
        break;
      }
    }
    if (init_output) {
      // If this is the first tile, we need to initialize the output.
      if (init_c.base() == c.base()) {
        // The input and accumulator were aliased to the same buffer, we don't
        // need to copy it.
        // TODO: Do we need to slice init_c first? Or maybe just fall through to
        // slinky::copy and make it optimize this case?
      } else {
        slinky::copy(init_c, c);
      }
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
    int reduction_idx = 0;
    for (int i = 0; i < a.rank;) {
      const slinky::dim& a_dim_i = a.dim(i);
      if (a_dim_i.empty()) {
        // The reduction is an empty reduction, so we are done after
        // initializing the output.
        return 0;
      } else if (op.k_dims[i + sliced]) {
        const slinky::dim& r_dim_i = reduction_bounds.dim(reduction_idx);
        assert(r_dim_i.stride() == 0);

        assert(a_dim_i.contains(r_dim_i));
        const slinky::index_t r_extent_i = r_dim_i.extent();
        assert(!a_dim_i.is_folded(r_dim_i.min(), r_dim_i.max()));
        if (r_extent_i == 1) {
          // Slice extent 1 dimensions, they don't affect the result.
        } else if (k1 * a.elem_size == a_dim_i.stride()) {
          k1 *= r_extent_i;
        } else if (a_stride_k2 * k2 == a_dim_i.stride()) {
          k2 *= r_extent_i;
        } else if (a_stride_k3 * k3 == a_dim_i.stride()) {
          k3 *= r_extent_i;
        } else if (k2 == 1) {
          k2 = r_extent_i;
          a_stride_k2 = a_dim_i.stride();
        } else if (k3 == 1) {
          k3 = r_extent_i;
          a_stride_k3 = a_dim_i.stride();
        } else {
          // This is a reduction dimension that is not handled by the kernel.
          if (op.keep_dims) {
            // Replace the existing dimension in c with a broadcast.
            c.mutable_dim(i) = r_dim_i;
          } else {
            // Add a new dimension for the broadcast in c.
            c.unslice(i, r_dim_i);
          }
          ++i;
          ++reduction_idx;
          continue;
        }
        // This is a reduction dimension that is handled by the kernel.
        a.slice(i, slinky::in_bounds{r_dim_i.min()});
        if (op.keep_dims) {
          // If op.keep_dims is true, that means the buffer has these
          // dimensions, but we don't want them there for dimensions we handle
          // with the kernel.
          c.slice(i);
        }
        ++sliced;
        ++reduction_idx;
      } else {
        const slinky::dim& c_dim_i = c.dim(i);
        const slinky::index_t c_extent_i = c_dim_i.extent();
        // Not a reduction dimension. If we haven't already found a dimension
        // for the kernel, give it this one.
        if (c_extent_i == 1) {
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
        a.slice(i, slinky::in_bounds{c_dim_i.min()});
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

slinky::raw_buffer_ptr make_reduce_identity(ynn_type type, int rank,
                                            ynn_reduce_operator op) {
  slinky::dim dims[YNN_MAX_TENSOR_RANK];
  slinky::raw_buffer value;
  value.rank = 0;
  value.elem_size = ynn::type_size_bytes(type);
  value.dims = dims;

  float value_f32[2];
  size_t n = 1;
  if (op == ynn_reduce_min_max) {
    value.rank = rank;
    for (int i = 0; i < rank - 1; ++i) {
      value.mutable_dim(i) = slinky::dim::broadcast();
    }
    value.mutable_dim(rank - 1) = slinky::dim(0, 1, value.elem_size);
    value_f32[0] = get_reduce_identity(ynn_reduce_min);
    value_f32[1] = get_reduce_identity(ynn_reduce_max);
    n = 2;
  } else {
    value_f32[0] = get_reduce_identity(op);
  }

  assert(type_size_bytes(type) <= sizeof(double));
  alignas(double) char storage[2 * sizeof(double)] = {0, };
  value.base = storage;
  convert_n(value_f32, n, type, value.base);
  return slinky::raw_buffer::make_copy(value);
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

  if (input_b_id != YNN_INVALID_VALUE_ID) {
    const ynn_value& b = subgraph.value(input_b_id);
    if (b.as_scalar_float() == get_reduce_identity(op)) {
      // This is the default value, using the default enables some fusions to
      // happen.
      input_b_id = YNN_INVALID_VALUE_ID;
    } else if (b.type != output.type) {
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
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    auto identity_buffer = slinky::buffer_expr::make_constant(
        runtime.globals.symbols, "identity",
        make_reduce_identity(output.type, output.rank(), op.op));

    ynn_runtime_value input_c;
    if (node.inputs[1] != YNN_INVALID_VALUE_ID) {
      input_c = runtime.value(node.inputs[1]);
    } else {
      input_c.buffer = identity_buffer;
    }

    output.make_buffer(runtime);

    // We want to be able to schedule producers for the reduction inside the
    // reduction loops. To be able to do this, we need the loops to be in the
    // slinky pipeline. To implement this, we make a dummy buffer that contains
    // the reduction dimensions as its dimensions, but with an element size and
    // stride of 0 (so it will not actually occupy any memory).
    // Both the output and this reduction buffer are outputs to the pipeline
    // stage, so slinky generates loops for both sets of dimensions. We can then
    // schedule these loops like any other dimensions.
    slinky::buffer_expr_ptr reduction_buffer = slinky::buffer_expr::make(
        runtime.globals.symbols, "reduction", op.k_dims.count(), 0);

    std::vector<slinky::var> output_dims;
    std::vector<slinky::var> reduction_dims;
    std::vector<slinky::var> all_dims;
    std::vector<slinky::expr> all_extents;
    slinky::box_expr a_bounds;
    slinky::box_expr a_crop;
    int reduction_dim = 0;
    for (int i = 0; i < input_a.rank(); ++i) {
      slinky::var dim_i = runtime.globals.make_dim(i);
      const slinky::expr& a_extent_i = input_a.extent(i);
      all_extents.push_back(a_extent_i);
      if (op.k_dims[i]) {
        slinky::var reduction_dim_i =
            runtime.globals.make_dim(reduction_dim, "k");
        all_dims.push_back(reduction_dim_i);
        reduction_dims.push_back(reduction_dim_i);

        a_bounds.push_back(slinky::point(reduction_dim_i));
        a_crop.push_back(slinky::min_extent(0, a_extent_i));
        if (op.keep_dims) {
          output_dims.push_back(dim_i);
        }

        // Set up the reduction buffer.
        reduction_buffer->dim(reduction_dim).bounds =
            slinky::min_extent(0, max(a_extent_i, 1));
        reduction_buffer->dim(reduction_dim).stride = 0;

        ++reduction_dim;
      } else {
        all_dims.push_back(dim_i);
        output_dims.push_back(dim_i);
        a_bounds.push_back(elementwise_bounds(dim_i, a_extent_i));
        a_crop.push_back({});
      }
    }

    slinky::box_expr c_bounds =
        make_elementwise_bounds(output_dims, input_c.extents);

    slinky::call_stmt::attributes attrs;
    attrs.name = node.to_string();
    // Allow the input_c and output to be computed in-place, which means we
    // don't need to initialize the accumulator.
    if (allow_in_place(input_c.id, output.id, *runtime.subgraph)) {
      attrs.allow_in_place = (1 << 1);
    }

    auto sched = runtime.make_schedule(all_dims, all_extents,
                                       input_a.buffer->elem_size());

    auto func = slinky::func::make(
        make_unary_reduce_impl(op, kernel),
        {{input_a.buffer, std::move(a_bounds), std::move(a_crop)},
         {input_c.buffer, std::move(c_bounds)}},
        {{output.buffer, std::move(output_dims)},
         {reduction_buffer, std::move(reduction_dims)}},
        std::move(attrs));

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
