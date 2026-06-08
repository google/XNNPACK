// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "ynnpack/base/algorithm.h"
#include "ynnpack/base/log.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/transpose/transpose.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/builder/pipeline.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

namespace ynn {

namespace {

auto make_transpose_impl(int elem_count,
                         const std::vector<int32_t>& permutation,
                         int input_packed_dim, int output_packed_dim) {
  return [elem_count, permutation, input_packed_dim, output_packed_dim](
             const slinky::raw_buffer& input,
             const slinky::raw_buffer& output) -> slinky::index_t {
    // Make a shallow copy of the input buffers. We need to be able to slice
    // dimensions from these buffers, and reorder the input dimensions.
    slinky::buffer<void, YNN_MAX_TENSOR_RANK> sliced_output = output;
    slinky::buffer<const void, YNN_MAX_TENSOR_RANK> sliced_input;
    sliced_input.rank = permutation.size();
    sliced_input.elem_size = input.elem_size;
    sliced_input.raw_buffer::base = input.base;

    // We need to find the dimension 0 of the input, and know where it is after
    // optimizing the copy.
    int input_dim0 = -1;
    int fuse_transpose[YNN_MAX_TENSOR_RANK];
    int fuse_batch[YNN_MAX_TENSOR_RANK];
    for (size_t d = 0; d < permutation.size(); ++d) {
      sliced_input.dims[d] = input.dim(permutation[d]);
      fuse_batch[d] = input_dim0 == -1 ? d : YNN_MAX_TENSOR_RANK;
      if (permutation[d] == input_packed_dim) {
        input_dim0 = d;
      }
      // Don't include input_dim0 in the set for fusion.
      fuse_transpose[d] = input_dim0 == -1 ? 0 : d;
    }

    // Fuse dimensions after the input dimension 0 first.
    slinky::fuse_contiguous_dims(fuse_batch, sliced_output, sliced_input);
    // Fuse dimensions that can be handled by a single loop, and update the
    // input dimension 0 accordingly. Due to the construction of `fusion_sets`
    // above, we're only going to fuse dimensions before `input_dim0`, so that
    // dimension moves by the number of dimensions fused.
    input_dim0 -= slinky::fuse_contiguous_dims(fuse_transpose, sliced_output,
                                               sliced_input);

    if (input_dim0 == 0 ||
        (elem_count == 1 && (input_dim0 <= 0 || sliced_output.rank < 2))) {
      // This transpose collapsed to a simple copy (one of the transposed
      // extents was 1?)
      slinky::copy(sliced_input, sliced_output);
      return 0;
    }

    const transpose_fn kernel =
        get_tiled_transpose(output.elem_size * 8 / elem_count);
    assert(kernel);

    const slinky::index_t m = sliced_output.dim(input_dim0).extent();
    bool input_dim0_is_packed = (permutation[0] == input_packed_dim);
    slinky::index_t input_n =
        sliced_input.dim(0).extent() * (input_dim0_is_packed ? elem_count : 1);
    bool output_dim0_is_packed = (output_packed_dim == 0);
    slinky::index_t output_n = sliced_output.dim(0).extent() *
                               (output_dim0_is_packed ? elem_count : 1);
    const slinky::index_t n = std::min(input_n, output_n);
    const slinky::index_t n_bytes_a = m * output.elem_size / elem_count;
    assert(is_contiguous(sliced_input.dim(input_dim0), output.elem_size));
    const slinky::index_t input_stride = sliced_input.dim(0).stride();
    assert(is_contiguous(sliced_output.dim(0), output.elem_size));
    const slinky::index_t output_stride =
        sliced_output.dim(input_dim0).stride();

    // Remove the transposed dimensions. These loops are inside the kernel.
    // We need to slice the input at the min of the output so we get the
    // correct pointers. `for_each_element` handles this for us for the
    // other dimensions. The order here is important because slicing dim0
    // would change the meaning of the input_dim0 index.
    sliced_input.slice(
        input_dim0,
        slinky::in_bounds{sliced_output.dim(input_dim0).min() / elem_count});
    sliced_input.slice(
        0, slinky::in_bounds{sliced_output.dim(0).min() * elem_count});
    sliced_output.slice({0, static_cast<size_t>(input_dim0)});

    slinky::for_each_element(
        [=, &kernel](void* out, const void* in) {
          kernel(m, n, n_bytes_a, input_stride, in, output_stride, out);
        },
        sliced_output, sliced_input);

    return 0;
  };
}

}  // namespace

void define_static_transpose(ynn_subgraph& subgraph, ynn_node& node,
                             std::vector<int32_t> permutation,
                             uint32_t input_id, uint32_t* output_id,
                             bool alias) {
  const ynn_value& input = subgraph.value(input_id);

  // Propagate shape.
  const int elem_count = type_element_count(input.type);
  std::vector<slinky::expr> output_extents(permutation.size());
  size_t first_non_trivial_dim = permutation.size();
  bool identity = permutation.size() == input.rank();
  for (size_t d = 0; d < output_extents.size(); ++d) {
    identity = identity && (permutation[d] == static_cast<int32_t>(d));
    slinky::expr input_extent = permutation[d] < input.rank()
                                    ? input.extents[permutation[d]]
                                    : slinky::expr{};
    if (input_extent.defined()) {
      first_non_trivial_dim = std::min(first_non_trivial_dim, d);
    }
    output_extents[d] = input_extent;
  }

  if (identity && *output_id == YNN_INVALID_VALUE_ID) {
    *output_id = input_id;
    return;
  }

  ynn_value& output = subgraph.get_output_value(output_id, input);
  output.extents = std::move(output_extents);

  // We can alias if we aren't rearranging the stride 1 dimension from the
  // input.
  alias = alias || permutation.empty() ||
          (elem_count == 1 && (first_non_trivial_dim >= permutation.size() ||
                               permutation[first_non_trivial_dim] == 0));

  node.inputs = {input_id};
  node.outputs = {output.id};
  node.op = ynn_node::static_transpose{std::move(permutation), alias};

  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const ynn_node::static_transpose& op =
        std::get<ynn_node::static_transpose>(node.op);
    const int input_id = node.inputs[0];
    const int output_id = node.outputs[0];
    const ynn_runtime_value& input = runtime.value(input_id);
    ynn_runtime_value& output = runtime.value(output_id);
    const size_t elem_count = type_element_count(output.type);

    output.make_buffer(runtime, input.buffer->elem_size());

    int rank = op.permutation.size();

    std::vector<slinky::var> output_dims = runtime.globals.make_dims(rank);
    slinky::box_expr bounds(input.rank(), slinky::point(0));
    for (int d = 0; d < rank; ++d) {
      if (op.permutation[d] < input.rank()) {
        bounds[op.permutation[d]] = slinky::point(output_dims[d]);
      }
    }

    int input_packed_dim = (type_element_count(input.type) != 1) ? 0 : -1;
    int output_packed_dim = (elem_count != 1) ? 0 : -1;

    if (elem_count != 1) {
      if (input_packed_dim >= 0 && any_n(rank, [&](int d) {
            return d != output_packed_dim &&
                   op.permutation[d] == input_packed_dim;
          })) {
        bounds[input_packed_dim] /= (int)elem_count;
      }
      if (output_packed_dim >= 0 &&
          output_packed_dim < static_cast<int>(op.permutation.size())) {
        int input_dim = op.permutation[output_packed_dim];
        if (input_dim != input_packed_dim && input_dim < input.rank()) {
          bounds[input_dim] = {
              bounds[input_dim].min * (int)elem_count,
              (bounds[input_dim].max + 1) * (int)elem_count - 1};
        }
      }
    }

    slinky::func f;
    auto sched = std::make_unique<scheduling_info>();
    if (op.alias) {
      f = slinky::func::make_copy({input.buffer, std::move(bounds)},
                                  {output.buffer, output_dims});
    } else {
      slinky::call_stmt::attributes attrs;
      attrs.name = "transpose";

      f = slinky::func::make(
          make_transpose_impl(elem_count, op.permutation, input_packed_dim,
                              output_packed_dim),
          {{input.buffer, std::move(bounds)}}, {{output.buffer, output_dims}},
          attrs);
    }

    f.user_data() = sched.get();
    runtime.scheduling_info_storage.push_back(std::move(sched));
    runtime.funcs.push_back(std::move(f));
    return ynn_status_success;
  };
}

void define_static_expand_dims(ynn_subgraph& subgraph, ynn_node& node,
                               uint32_t input_id, uint32_t* output_id,
                               const axes_set& new_axes) {
  const ynn_value& input = subgraph.value(input_id);

  // This is implemented by a transpose that is an identity permutation, except
  // with the new dimensions inserted.
  std::vector<int32_t> permutation(input.rank() + new_axes.count());
  int dim = 0;
  for (int i = 0; i < permutation.size(); ++i) {
    permutation[i] = new_axes[i] ? input.rank() : dim++;
  }

  define_static_transpose(subgraph, node, std::move(permutation), input_id,
                          output_id, /*alias=*/true);
}

std::optional<axes_set> get_static_expand_dims_axes(
    const ynn_node::static_transpose& op, int input_rank) {
  axes_set axes;
  int next_input_dim = 0;
  for (size_t i = 0; i < op.permutation.size(); ++i) {
    if (op.permutation[i] < 0 || op.permutation[i] >= input_rank) {
      axes[i] = true;
    } else if (op.permutation[i] == next_input_dim) {
      next_input_dim++;
    } else {
      return std::nullopt;
    }
  }
  if (next_input_dim != input_rank) {
    return std::nullopt;
  }
  return axes;
}

extern "C" {

ynn_status ynn_define_static_transpose(ynn_subgraph_t subgraph, size_t rank,
                                       const int32_t* permutation,
                                       uint32_t input_id, uint32_t* output_id,
                                       uint32_t flags) {
  // Validate arguments.
  YNN_RETURN_IF_ERROR(validate_subgraph("static_transpose", subgraph));
  YNN_RETURN_IF_ERROR(validate_input_tensor("static_transpose", subgraph,
                                            "input_id", input_id));
  YNN_RETURN_IF_ERROR(validate_output_tensor("static_transpose", subgraph,
                                             "output_id", output_id));
  if (permutation == nullptr && rank > 0) {
    YNN_LOG_ERROR() << "For node `static_transpose`, permutation must be "
                       "non-null for rank > 0";
    return ynn_status_invalid_parameter;
  }
  YNN_RETURN_IF_ERROR(validate_rank("static_transpose", "output", rank));

  // Rewrite the permutation to be slinky dimensions.
  const ynn_value& input = subgraph->value(input_id);
  std::vector<int32_t> op_permutation(rank);
  for (size_t i = 0; i < rank; ++i) {
    op_permutation[i] = axis_to_slinky_dim(input.rank(), permutation[i]);
    if (op_permutation[i] < 0 || op_permutation[i] >= input.rank()) {
      // This means we insert a new dimension of extent 1.
      op_permutation[i] = input.rank();
    }
  }
  std::reverse(op_permutation.begin(), op_permutation.end());

  ynn_node node;
  define_static_transpose(*subgraph, node, std::move(op_permutation), input_id,
                          output_id, flags);
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

ynn_status ynn_define_static_expand_dims(ynn_subgraph_t subgraph,
                                         size_t num_new_axes,
                                         const int32_t* new_axes,
                                         uint32_t input_id, uint32_t* output_id,
                                         uint32_t flags) {
  // Validate arguments.
  YNN_RETURN_IF_ERROR(validate_subgraph("static_expand_dims", subgraph));
  YNN_RETURN_IF_ERROR(validate_input_tensor("static_expand_dims", subgraph,
                                            "input_id", input_id));
  YNN_RETURN_IF_ERROR(validate_output_tensor("static_expand_dims", subgraph,
                                             "output_id", output_id));

  const ynn_value& input = subgraph->value(input_id);

  const int new_rank = input.rank() + num_new_axes;
  YNN_RETURN_IF_ERROR(validate_rank("static_expand_dims", "output", new_rank));
  ynn::axes_set axes;
  for (size_t i = 0; i < num_new_axes; ++i) {
    YNN_RETURN_IF_ERROR(
        validate_axis("static_expand_dims", "output", new_rank, new_axes[i]));
    axes[axis_to_slinky_dim(new_rank, new_axes[i])] = true;
  }

  ynn_node node;
  define_static_expand_dims(*subgraph, node, input_id, output_id, axes);
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

}  // extern "C"

}  // namespace ynn
