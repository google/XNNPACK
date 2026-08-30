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
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "ynnpack/base/algorithm.h"
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/log.h"
#include "ynnpack/base/span.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/transpose/transpose.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/builder/pipeline.h"
#include "slinky/builder/simplify.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

namespace ynn {

namespace {

auto make_transpose_impl(int elem_count, std::vector<int32_t> permutation) {
  return [elem_count, permutation](
             const slinky::raw_buffer& input,
             const slinky::raw_buffer& output) -> slinky::index_t {
    // Make a shallow copy of the input buffers. We need to be able to slice
    // dimensions from these buffers, and reorder the input dimensions.
    slinky::buffer<void, max_tensor_rank> sliced_output = output;
    slinky::buffer<const void, max_tensor_rank> sliced_input;
    sliced_input.rank = permutation.size();
    sliced_input.elem_size = input.elem_size;
    sliced_input.raw_buffer::base = input.base;

    for (size_t d = 0; d < permutation.size(); ++d) {
      sliced_input.dims[d] = input.dim(permutation[d]);
    }

    if (elem_count != 1) {
      // TODO: b/532524411 - We should be able to optimize dims when we have
      // non-byte types too.
    } else {
      // Sort and fuse dimensions
      slinky::optimize_dims(sliced_output, sliced_input);
    }

    // Fold copies of contiguous dimensions into the elem_size.
    slinky::index_t elem_size = sliced_output.elem_size;
    int sliced_elem_count = elem_count;
    while (sliced_output.rank > 0 && (elem_count == 1 || permutation[0] == 0) &&
           is_contiguous(sliced_output.dim(0), elem_size) &&
           is_contiguous(sliced_input.dim(0), elem_size) &&
           // Only fold dims if their extent spans the physical stride to the
           // next dim.
           (sliced_output.rank == 1 ||
            (sliced_output.dim(0).extent() * elem_size ==
                 sliced_output.dim(1).stride() &&
             sliced_input.dim(0).extent() * elem_size ==
                 sliced_input.dim(1).stride()))) {
      elem_size *= sliced_output.dim(0).extent();
      sliced_input.slice(0, slinky::in_bounds{sliced_output.dim(0).min()});
      sliced_output.slice(0);
      sliced_elem_count = 1;
    }

    const transpose_fn kernel =
        get_tiled_transpose(elem_size * 8 / sliced_elem_count);
    assert(kernel);

    // Find the contiguous dimension in the input, which is the dimension we
    // need to handle with the kernel.
    size_t input_dim0 = sliced_input.rank;
    for (size_t d = 1; d < permutation.size(); ++d) {
      if (is_contiguous(sliced_input.dim(d), elem_size)) {
        if (d < permutation.size() &&
            (input_dim0 >= permutation.size() ||
             permutation[d] < permutation[input_dim0])) {
          input_dim0 = d;
        }
      }
    }

    const slinky::dim& output_m = sliced_output.dim(input_dim0);
    const slinky::dim& output_n = sliced_output.dim(0);
    const slinky::index_t m = output_m.extent();
    const slinky::index_t n = output_n.extent() * sliced_elem_count;
    const slinky::index_t n_bytes_a =
        ceil_div<slinky::index_t>(m * elem_size, sliced_elem_count);
    const slinky::index_t input_stride = sliced_input.dim(0).stride();
    const slinky::index_t output_stride = output_m.stride();

    // Remove the transposed dimensions. These loops are inside the kernel.
    // We need to slice the input at the min of the output so we get the
    // correct pointers. `for_each_element` handles this for us for the
    // other dimensions. The order here is important because slicing dim0
    // would change the meaning of the input_dim0 index.
    assert(output_m.min() % sliced_elem_count == 0);
    sliced_input.slice(input_dim0,
                       slinky::in_bounds{output_m.min() / sliced_elem_count});
    sliced_input.slice(0,
                       slinky::in_bounds{output_n.min() * sliced_elem_count});
    sliced_output.slice({0, input_dim0});

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
  bool identity = permutation.size() == input.rank();
  for (size_t d = 0; d < output_extents.size(); ++d) {
    identity = identity && (permutation[d] == static_cast<int32_t>(d));
    slinky::expr input_extent = permutation[d] < input.rank()
                                    ? input.extents[permutation[d]]
                                    : slinky::expr{};
    output_extents[d] = input_extent;
  }
  if (elem_count != 1 && !output_extents.empty() &&
      output_extents[0].defined()) {
    // And convert back to a physical shape after converting to a logical
    // shape above. This could fail if the user transposes a dimension with an
    // extent that is not aligned to `elem_count`.
    node.checks.push_back(ynn_node::check{
        output_extents[0] % elem_count == 0,
        {"For node 'static_transpose', dimension 0 extent (", output_extents[0],
         ") of ", ynn_node::output_idx{0},
         " is not aligned to an instance of type ", to_string(input.type)},
    });
  }

  if (identity && *output_id == YNN_INVALID_VALUE_ID) {
    *output_id = input_id;
    return;
  }

  ynn_value& output = subgraph.get_output_value(output_id, input);
  output.extents = std::move(output_extents);

  // We can alias if we aren't rearranging the stride 1 dimension from the
  // input.
  size_t first_non_trivial_output_dim = first_non_trivial_dim(output.extents);
  size_t first_non_trivial_input_dim = first_non_trivial_dim(input.extents);
  alias =
      alias || permutation.empty() ||
      first_non_trivial_output_dim >= permutation.size() ||
      first_non_trivial_input_dim >= input.rank() ||
      permutation[first_non_trivial_output_dim] == first_non_trivial_input_dim;

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
    const int elem_count = type_element_count(output.type);

    output.make_buffer(runtime, input.buffer->elem_size());

    int rank = op.permutation.size();

    std::vector<slinky::var> output_dims = runtime.globals.make_dims(rank);
    slinky::box_expr bounds(input.rank(), slinky::point(0));
    for (int d = 0; d < rank; ++d) {
      if (op.permutation[d] < input.rank()) {
        bounds[op.permutation[d]] = slinky::point(output_dims[d]);
      }
    }

    if (elem_count != 1) {
      if (any_n(rank, [&](int d) { return d > 0 && op.permutation[d] == 0; })) {
        // We're loading the packed dimensions with an index from a non-packed
        // dimension, adjust for the number of elements.
        bounds[0] /= elem_count;
      }
      if (rank > 0 && op.permutation[0] > 0 &&
          op.permutation[0] < input.rank()) {
        // We're loading a non-packed dimension with an index from the packed
        // dimension, adjust for the number of elements.
        bounds[op.permutation[0]] *= elem_count;
      }
    }

    slinky::func f;
    std::unique_ptr<scheduling_info> sched;
    if (op.alias) {
      f = slinky::func::make_copy({input.buffer, std::move(bounds)},
                                  {output.buffer, output_dims});
    } else {
      slinky::call_stmt::attributes attrs;
      attrs.name = "transpose";

      f = slinky::func::make(make_transpose_impl(elem_count, op.permutation),
                             {{input.buffer, std::move(bounds)}},
                             {{output.buffer, output_dims}}, attrs);
      // Tile the transpose so it can be parallelized and stays cache
      // friendly. A tile is read as contiguous rows of the input and written
      // as contiguous rows of the output, so both sides get reservation
      // floors (alignments) targeting rows of ~1KB: without its floor,
      // either side can be starved of tile area by the other, degrading the
      // tile to strided scalar-ish access on that side. Each side reserves
      // along its dimensions in memory order: the innermost dimension
      // reserves the whole row target, and when its extent is smaller than
      // that, the following dimensions extend the contiguous run and reserve
      // the factor still missing (e.g. transposing 262144x256x4 int8 to
      // 256x4x262144: input rows of 4 elements are extended to 4x256).
      constexpr slinky::index_t row_target_bytes = 1024;
      const size_t size_bytes =
          std::max<size_t>(1, type_size_bytes(output.type));
      std::vector<slinky::expr> extents = output.physical_extents();
      std::vector<slinky::expr> alignments(rank);

      auto reserve = [&](int d, const slinky::expr& floor) {
        slinky::expr a =
            slinky::simplify(slinky::min(slinky::max(1, floor), extents[d]));
        if (alignments[d].defined()) {
          alignments[d] = slinky::simplify(slinky::max(alignments[d], a));
        } else {
          alignments[d] = a;
        }
        return a;
      };

      // Output rows. Dimension 0 of the output buffer is counted in physical
      // (packed) units of size_bytes each, so the row target is too.
      const slinky::index_t out_row =
          std::max<slinky::index_t>(1, row_target_bytes / size_bytes);
      slinky::expr out_run = 1;
      for (int d = 0; d < rank && extents[d].defined(); ++d) {
        slinky::expr a =
            reserve(d, slinky::ceil_div(slinky::expr(out_row), out_run));
        out_run = slinky::simplify(out_run * a);
      }

      // Input rows, reserved on the output dimension each input dimension
      // maps to. These are counted in logical elements.
      const slinky::index_t in_row = std::max<slinky::index_t>(
          1, row_target_bytes * type_element_count(output.type) / size_bytes);
      slinky::expr in_run = 1;
      for (int k = 0; k < input.rank(); ++k) {
        int d = 0;
        while (d < rank && op.permutation[d] != k) d++;
        if (d >= rank || !extents[d].defined()) continue;
        slinky::expr a =
            reserve(d, slinky::ceil_div(slinky::expr(in_row), in_run));
        if (d == 0) {
          // Output dimension 0 is in physical units; the input run it
          // contributes is elem_count logical elements per unit.
          a = a * type_element_count(output.type);
        }
        in_run = slinky::simplify(in_run * a);
      }
      std::vector<slinky::expr> splits = make_split_factors(
          runtime.globals, extents, type_size_bytes(output.type),
          /*given_splits=*/{}, /*loop_order=*/{}, alignments);
      sched = runtime.make_schedule(output_dims, extents, splits);
    }
    if (!sched) {
      sched = std::make_unique<scheduling_info>();
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

ynn_status ynn_define_static_transpose(ynn_subgraph_t subgraph, size_t num_axes,
                                       const int32_t* axes,
                                       uint32_t input_id, uint32_t* output_id,
                                       uint32_t flags) {
  // Validate arguments.
  YNN_RETURN_IF_ERROR(validate_subgraph("static_transpose", subgraph));
  YNN_RETURN_IF_ERROR(validate_input_tensor("static_transpose", subgraph,
                                            "input_id", input_id));
  YNN_RETURN_IF_ERROR(validate_output_tensor("static_transpose", subgraph,
                                             "output_id", output_id));
  if (axes == nullptr && num_axes > 0) {
    YNN_LOG_ERROR() << "For node `static_transpose`, permutation must be "
                       "non-null for rank > 0";
    return ynn_status_invalid_parameter;
  }
  const ynn_value& input = subgraph->value(input_id);

  std::vector<int32_t> internal_axes;
  internal_axes.reserve(num_axes);
  for (size_t i = 0; i < num_axes; ++i) {
    int32_t axis = axis_to_slinky_dim(input.rank(), axes[i]);
    if (axis < 0 || axis >= input.rank()) {
      if (flags & YNN_NODE_FLAG_KEEP_DIMS) {
        YNN_LOG_ERROR() << "For node `static_transpose`, axis "
                        << axes[i] << " is beyond the rank "
                        << input.rank() << " of the input";
        return ynn_status_invalid_parameter;
      } else {
        // This means we insert a new dimension of extent 1.
        axis = input.rank();
      }
    }
    internal_axes.push_back(axis);
  }

  std::vector<int32_t> op_permutation;
  if (flags & YNN_NODE_FLAG_KEEP_DIMS) {
    std::vector<int32_t> positions = internal_axes;
    std::sort(positions.begin(), positions.end(), std::greater<int32_t>());

    op_permutation.resize(input.rank());
    std::iota(op_permutation.begin(), op_permutation.end(), 0);
    for (size_t k = 0; k < num_axes; ++k) {
      op_permutation[positions[k]] = internal_axes[k];
    }
  } else {
    YNN_RETURN_IF_ERROR(validate_rank("static_transpose", "output", num_axes));
    op_permutation = std::move(internal_axes);
    std::reverse(op_permutation.begin(), op_permutation.end());
  }

  ynn_node node;
  define_static_transpose(*subgraph, node, std::move(op_permutation), input_id,
                          output_id, /*alias=*/false);
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
