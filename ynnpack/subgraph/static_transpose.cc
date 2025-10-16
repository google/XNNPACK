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
#include <utility>
#include <vector>

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

auto make_transpose_impl(int rank, size_t elem_count,
                         std::vector<int32_t> permutation) {
  return [rank, elem_count, permutation](
             const slinky::buffer<const void>& input,
             const slinky::buffer<void>& output) -> slinky::index_t {
    // Make a shallow copy of the input buffers. We need to be able to slice
    // dimensions from these buffers, and reorder the input dimensions.
    slinky::buffer<void, YNN_MAX_TENSOR_RANK> sliced_output = output;
    slinky::buffer<const void, YNN_MAX_TENSOR_RANK> sliced_input = input;

    // We need to find the dimension 0 of the input, and know where it is after
    // optimizing the copy.
    int input_dim0 = -1;
    int fuse_transpose[YNN_MAX_TENSOR_RANK];
    int fuse_batch[YNN_MAX_TENSOR_RANK];
    for (int d = 0; d < rank; ++d) {
      sliced_input.dim(d) = input.dim(permutation[d]);
      fuse_batch[d] = input_dim0 == -1 ? d : YNN_MAX_TENSOR_RANK;
      if (permutation[d] == 0) {
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

    if (input_dim0 <= 0 || sliced_output.rank < 2) {
      // This transpose collapsed to a simple copy (one of the transposed
      // extents was 1?)
      slinky::copy(sliced_input, sliced_output);
      return 0;
    }

    const transpose_kernel_fn ukernel =
        get_transpose_kernel(output.elem_size * 8 / elem_count);
    assert(ukernel);

    const slinky::index_t m = sliced_output.dim(input_dim0).extent();
    const slinky::index_t n = sliced_output.dim(0).extent() * elem_count;
    const slinky::index_t n_bytes_a = m * output.elem_size;
    assert(sliced_input.dim(input_dim0).stride() == output.elem_size);
    const slinky::index_t input_stride = sliced_input.dim(0).stride();
    assert(sliced_output.dim(0).stride() == output.elem_size);
    const slinky::index_t output_stride =
        sliced_output.dim(input_dim0).stride();

    // Remove the transposed dimensions. These loops are inside the ukernel.
    // We need to slice the input at the min of the output so we get the
    // correct pointers. `for_each_element` handles this for us for the
    // other dimensions. The order here is important because slicing dim0
    // would change the meaning of the input_dim0 index.
    sliced_input.slice(input_dim0,
                       sliced_output.dim(input_dim0).min() / elem_count);
    sliced_input.slice(0, sliced_output.dim(0).min() * elem_count);
    sliced_output.slice({0, static_cast<size_t>(input_dim0)});

    slinky::for_each_element(
        [=](void* out, const void* in) {
          ukernel(m, n, n_bytes_a, input_stride, in, output_stride, out);
        },
        sliced_output, sliced_input);

    return 0;
  };
}

}  // namespace

ynn_status define_static_transpose(ynn_subgraph_t subgraph,
                                   std::vector<int32_t> permutation,
                                   uint32_t input_id, uint32_t* output_id,
                                   bool alias) {
  // Validate arguments.
  assert(subgraph);
  assert(subgraph->is_valid_value(input_id));
  const ynn_value& input = subgraph->value(input_id);

  ynn_value& output = subgraph->get_output_value(output_id, input);

  ynn_node node;
  node.inputs = {input_id};
  node.outputs = {output.id};

  // Propagate shape.
  const int elem_count = type_element_count(output.type);
  output.extents.resize(permutation.size());
  for (int d = 0; d < output.rank(); ++d) {
    slinky::expr input_extent = input.extents[permutation[d]];
    if (permutation[d] == 0 && elem_count != 1) {
      // The extents are physical shapes, we need to convert to logical shapes
      // when we transpose the dimensions.
      input_extent *= elem_count;
    }
    output.extents[d] = input_extent;
  }
  if (elem_count != 1) {
    // And convert back to a physical shape after converting to a logical
    // shape above. This could fail if the user transposes a dimension with an
    // extent that is not aligned to `elem_count`.
    node.checks.push_back(ynn_node::check{
        output.extents[0] % elem_count == 0,
        {"In node 'static_transpose', dimension 0 extent (", output.extents[0],
         ") of ", ynn_node::output_idx{0},
         " is not aligned to an instance of type ", to_string(output.type)},
    });
    output.extents[0] /= elem_count;
  }

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

    std::vector<slinky::var> input_dims = make_dims(rank, runtime.symbols);
    std::vector<slinky::var> output_dims(input_dims);
    slinky::func::input func_input{
        input.buffer, make_elementwise_bounds(input_dims, input.extents)};

    auto sched = std::make_unique<scheduling_info>();

    for (int d = 0; d < rank; ++d) {
      output_dims[d] = input_dims[op.permutation[d]];
    }

    const bool transpose_stride_1 = rank > 1 && (op.permutation[0] != 0);
    if (transpose_stride_1) {
      // We're loading the packed dimensions with an index from a non-packed
      // dimension, adjust for the number of elements.
      func_input.bounds[0] /= elem_count;
    }

    slinky::func f;
    std::vector<slinky::index_t> schedule_alignments;
    if (op.alias) {
      f = slinky::func::make_copy(std::move(func_input),
                                  {output.buffer, output_dims});
      sched->force_root = true;
    } else {
      slinky::call_stmt::attributes attrs;
      attrs.name = "transpose";

      f = slinky::func::make(
          make_transpose_impl(rank, elem_count, op.permutation),
          {std::move(func_input)}, {{output.buffer, output_dims}}, attrs);
    }

    f.user_data() = sched.get();
    runtime.scheduling_info_storage.push_back(std::move(sched));
    runtime.funcs.push_back(std::move(f));
    return ynn_status_success;
  };
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

extern "C" {

ynn_status ynn_define_static_transpose(ynn_subgraph_t subgraph, size_t rank,
                                       const int32_t* permutation,
                                       uint32_t input_id, uint32_t* output_id,
                                       uint32_t flags) {
  // Rewrite the permutation to be slinky dimensions.
  assert(subgraph);
  assert(subgraph->is_valid_value(input_id));
  const ynn_value& input = subgraph->value(input_id);
  std::vector<int32_t> op_permutation(rank);
  for (size_t i = 0; i < rank; ++i) {
    op_permutation[i] = axis_to_slinky_dim(input.rank(), permutation[i]);
  }
  std::reverse(op_permutation.begin(), op_permutation.end());

  return define_static_transpose(subgraph, std::move(op_permutation), input_id,
                                 output_id);
}

}  // extern "C"

}  // namespace ynn
