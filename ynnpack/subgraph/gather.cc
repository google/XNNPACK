// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/gather.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <utility>
#include <vector>

#include "ynnpack/base/log.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/lut/lut.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/builder/pipeline.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

namespace ynn {

namespace {

// Call a lut kernel.
auto make_lut_impl(lut_kernel_fn kernel) {
  return [kernel](slinky::raw_buffer lut, slinky::raw_buffer a,
                  slinky::raw_buffer x) -> slinky::index_t {
    assert(is_contiguous(x.dim(0), x.elem_size));

    slinky::dim x_dim = ynn::slice_dim0(x);
    if (x_dim.empty()) {
      return 0;
    }
    slinky::dim a_dim = ynn::slice_dim0(a, slinky::in_bounds{x_dim.min()});

    assert(is_contiguous(a_dim, a.elem_size));
    assert(is_contiguous(x_dim, x.elem_size));
    (void)a_dim;

    const slinky::index_t x_n_extent = x_dim.extent();

    // Slice the lookup dimension of lut (dim 0).
    const size_t lut_size = lut.dim(0).extent();
    lut.slice(0, 0);

    bool success = true;
    slinky::for_each_element(
        [=, &success](void* x, const void* a, const void* lut_ptr) {
          success = success && kernel(x_n_extent, a, lut_size, lut_ptr, x);
        },
        x, a, lut);
    return success ? 0 : 1;
  };
}

slinky::index_t read_index_value(const void* ptr, ynn_type type) {
  switch (type) {
    case ynn_type_int8:
      return *reinterpret_cast<const int8_t*>(ptr);
    case ynn_type_uint8:
      return *reinterpret_cast<const uint8_t*>(ptr);
    case ynn_type_int32:
      return *reinterpret_cast<const int32_t*>(ptr);
    default:
      assert(false && "Unsupported index type");
      return 0;
  }
}

struct GatherMapping {
  size_t s_out_start = 0;
  size_t lookup_rank = 0;
  std::vector<size_t> preserved_input_dims;

  int output_to_input(size_t d) const {
    if (d < s_out_start) {
      return preserved_input_dims[d];
    } else if (d < s_out_start + lookup_rank) {
      return -1;
    } else {
      return preserved_input_dims[d - lookup_rank];
    }
  }
};

GatherMapping compute_gather_mapping(
    size_t input_rank, size_t output_rank, size_t lookup_rank,
    const std::vector<int32_t>& gathered_axes) {
  GatherMapping mapping;
  mapping.lookup_rank = lookup_rank;

  for (size_t d = 0; d < input_rank; ++d) {
    if (std::find(gathered_axes.begin(), gathered_axes.end(), d) ==
        gathered_axes.end()) {
      mapping.preserved_input_dims.push_back(d);
    }
  }

  int32_t max_s_axis =
      gathered_axes.empty()
          ? 0
          : *std::max_element(gathered_axes.begin(), gathered_axes.end());
  int32_t u_a_0 = input_rank - 1 - max_s_axis;
  mapping.s_out_start = output_rank - lookup_rank - u_a_0;

  return mapping;
}

// Creates the runtime JIT execution lambda for the gather operation.
//
// Arguments:
// - gathered_axes: Slinky indices of the input axes being gathered.
// - input_rank: Compile-time rank of the input tensor.
// - output_rank: Compile-time rank of the output tensor.
// - index_type: Data type of the index tensor.
// - s_coord_dim: Slinky index of the coordinate dimension in the index tensor
//   (size M), or -1 if no coordinate dimension exists (when M = 1).
//
// Compile-time Rank vs. Runtime Rank:
// Slinky's compiler may optimize intermediate buffers by collapsing size-1
// dimensions. This can cause the runtime buffer `input` to have a smaller rank
// than `input_rank`. If this happens, we unsqueeze `input` back to `input_rank`
// by inserting broadcast dimensions at the collapsed indices (tracked by
// `preserved_input_dims`). This ensures the gather indexing math remains
// correct relative to the compile-time `gathered_axes`.
//
// Coordinate Dimension Slicing:
// If the index tensor contains a coordinate dimension (s_coord_dim != -1), we
// slice it out in the JIT lambda before loop execution to get the pure lookup
// shape. We preserve its dimension descriptor (`axis_index_dim`) to read the
// individual coordinate values during the gather loop.
auto make_gather_impl(std::vector<int32_t> gathered_axes, size_t input_rank,
                      size_t output_rank, size_t compile_index_rank,
                      ynn_type index_type, int32_t s_coord_dim,
                      std::vector<bool> input_dims_collapsible,
                      std::vector<bool> index_dims_collapsible) {
  bool has_coordinate_dim = (s_coord_dim != -1);
  size_t compile_lookup_rank =
      has_coordinate_dim ? compile_index_rank - 1 : compile_index_rank;

  GatherMapping mapping = compute_gather_mapping(
      input_rank, output_rank, compile_lookup_rank, gathered_axes);

  bool is_aligned =
      (compile_lookup_rank == output_rank) && (input_rank == output_rank);
  std::vector<int> output_to_input_dim(output_rank);
  size_t compile_s_out_start = 0;

  if (is_aligned) {
    for (size_t d = 0; d < output_rank; ++d) {
      if (std::find(gathered_axes.begin(), gathered_axes.end(), d) ==
          gathered_axes.end()) {
        output_to_input_dim[d] = d;
      } else {
        output_to_input_dim[d] = -1;
      }
    }
  } else {
    compile_s_out_start = mapping.s_out_start;
    for (size_t d = 0; d < output_rank; ++d) {
      output_to_input_dim[d] = mapping.output_to_input(d);
    }
  }

  return
      [gathered_axes = std::move(gathered_axes), input_rank, output_rank,
       compile_lookup_rank, compile_s_out_start,
       output_to_input_dim = std::move(output_to_input_dim), index_type,
       s_coord_dim, input_dims_collapsible = std::move(input_dims_collapsible),
       index_dims_collapsible = std::move(index_dims_collapsible)](
          slinky::buffer<const void, max_tensor_rank> input,
          slinky::buffer<const void, max_tensor_rank> index,
          slinky::buffer<void, max_tensor_rank> output) -> slinky::index_t {
        if (input.rank < input_rank) {
          slinky::buffer<const void, max_tensor_rank> unsqueezed_input;
          unsqueezed_input.raw_buffer::base = const_cast<void*>(input.base());
          unsqueezed_input.elem_size = input.elem_size;
          unsqueezed_input.rank = input_rank;
          for (size_t d = 0; d < input_rank; ++d) {
            unsqueezed_input.mutable_dim(d) = slinky::dim::broadcast();
          }
          std::vector<size_t> runtime_to_compile(input.rank);
          size_t i = 0;
          for (size_t d = 0; d < input_rank; ++d) {
            if (i >= input.rank) break;
            if (input_dims_collapsible[d]) {
              if (input.dim(i).extent() > 1) {
                continue;
              }
              runtime_to_compile[i] = d;
              ++i;
            } else {
              runtime_to_compile[i] = d;
              ++i;
            }
          }
          assert(i == input.rank);
          for (size_t i = 0; i < input.rank; ++i) {
            unsqueezed_input.mutable_dim(runtime_to_compile[i]) = input.dim(i);
          }
          input = unsqueezed_input;
        }
        bool has_coordinate_dim = (s_coord_dim != -1);
        slinky::dim axis_index_dim = has_coordinate_dim
                                         ? index.dim(s_coord_dim)
                                         : slinky::dim::broadcast();
        if (has_coordinate_dim) {
          index.slice(s_coord_dim);
        }

        // 1. Pad index to compile_lookup_rank
        if (index.rank < compile_lookup_rank) {
          slinky::buffer<const void, max_tensor_rank> padded_index;
          padded_index.raw_buffer::base = const_cast<void*>(index.base());
          padded_index.elem_size = index.elem_size;
          padded_index.rank = compile_lookup_rank;
          for (size_t d = 0; d < compile_lookup_rank; ++d) {
            padded_index.mutable_dim(d) = slinky::dim::broadcast();
          }
          std::vector<size_t> runtime_to_compile(index.rank);
          size_t i = 0;
          for (size_t d = 0; d < compile_lookup_rank; ++d) {
            if (i >= index.rank) break;
            if (index_dims_collapsible[d]) {
              if (index.dim(i).extent() > 1) {
                continue;
              }
              runtime_to_compile[i] = d;
              ++i;
            } else {
              runtime_to_compile[i] = d;
              ++i;
            }
          }
          assert(i == index.rank);
          for (size_t i = 0; i < index.rank; ++i) {
            padded_index.mutable_dim(runtime_to_compile[i]) = index.dim(i);
          }
          index = padded_index;
        }

        // 2. Unsqueeze index to output_rank if not aligned
        if (compile_lookup_rank != output_rank) {
          if (index.rank < output_rank) {
            slinky::buffer<const void, max_tensor_rank> unsqueezed_index;
            unsqueezed_index.raw_buffer::base = const_cast<void*>(index.base());
            unsqueezed_index.elem_size = index.elem_size;
            unsqueezed_index.rank = output_rank;
            for (size_t d = 0; d < output_rank; ++d) {
              unsqueezed_index.mutable_dim(d) = slinky::dim::broadcast();
            }
            for (size_t i = 0; i < compile_lookup_rank; ++i) {
              size_t s_out = compile_s_out_start + i;
              unsqueezed_index.mutable_dim(s_out) = index.dim(i);
            }
            index = unsqueezed_index;
          }
        }

        // We're going to address the gathered dimensions separately from the
        // loop over the output.
        slinky::dim gathered_input_dims[max_tensor_rank];
        size_t num_gathered_dims = gathered_axes.size();
        for (size_t i = 0; i < num_gathered_dims; ++i) {
          gathered_input_dims[i] = input.dim(gathered_axes[i]);
          input.mutable_dim(gathered_axes[i]).set_stride(0);
        }

        // We need two sets of buffers: one for defining an outer loop, and one
        // to define how we can slinky::copy inside that outer loop.
        slinky::buffer<void, max_tensor_rank> output_slice = output;
        slinky::buffer<const void, max_tensor_rank> input_slice = input;

        // Slice input_slice
        std::vector<size_t> input_dims_to_remove(gathered_axes.begin(),
                                                 gathered_axes.end());
        for (size_t i = 0; i < output_rank; ++i) {
          if (!index.dim(i).is_broadcast()) {
            if (output_to_input_dim[i] != -1) {
              input_dims_to_remove.push_back(output_to_input_dim[i]);
            }
          }
        }
        std::sort(input_dims_to_remove.begin(), input_dims_to_remove.end(),
                  std::greater<size_t>());
        input_dims_to_remove.erase(std::unique(input_dims_to_remove.begin(),
                                               input_dims_to_remove.end()),
                                   input_dims_to_remove.end());
        for (size_t d : input_dims_to_remove) {
          input_slice.slice(d);
        }

        // Slice output_slice
        std::vector<size_t> output_dims_to_remove;
        for (size_t i = 0; i < output_rank; ++i) {
          if (!index.dim(i).is_broadcast()) {
            output_dims_to_remove.push_back(i);
          }
        }
        std::sort(output_dims_to_remove.begin(), output_dims_to_remove.end(),
                  std::greater<size_t>());
        for (size_t d : output_dims_to_remove) {
          output_slice.slice(d);
        }

        for (int i = output.rank - 1; i >= 0; --i) {
          if (index.dim(i).is_broadcast()) {
            if (output_to_input_dim[i] != -1) {
              input.slice(output_to_input_dim[i]);
              output.slice(i);
              index.slice(i);
            }
          }
        }

        bool error = false;
        slinky::for_each_element(
            [=, &error, &output_slice, &input_slice](void* output_ptr,
                                                     const void* index_ptr,
                                                     const void* input_ptr) {
              for (size_t j = 0; j < num_gathered_dims; ++j) {
                slinky::index_t idx = read_index_value(
                    slinky::offset_bytes(index_ptr,
                                         axis_index_dim.flat_offset_bytes(j)),
                    index_type);

                if (!gathered_input_dims[j].contains(idx)) {
                  error = true;
                  return;
                }
                input_ptr = slinky::offset_bytes(
                    input_ptr, gathered_input_dims[j].flat_offset_bytes(idx));
              }

              output_slice.raw_buffer::base = output_ptr;
              input_slice.raw_buffer::base = const_cast<void*>(input_ptr);

              slinky::copy(input_slice, output_slice);
            },
            output, index, input);

        return error;
      };
}

}  // namespace

void define_gather(ynn_subgraph& subgraph, ynn_node& node,
                   std::vector<int32_t> axes, size_t output_rank,
                   uint32_t input_id, uint32_t index_id, uint32_t& output_id) {
  const ynn_value& input = subgraph.value(input_id);
  const ynn_value& index = subgraph.value(index_id);

  ynn_value& output = subgraph.get_output_value(&output_id, input);

  node.inputs = {input_id, index_id};
  node.outputs = {output_id};

  std::vector<int32_t> sorted_axes = axes;
  std::sort(sorted_axes.begin(), sorted_axes.end());
  // M is the number of axes being gathered (e.g. M=1 for single-axis gather).
  size_t M = sorted_axes.size();
  // R_idx is the rank of the index tensor.
  size_t R_idx = index.rank();
  bool has_coordinate_dim = (output_rank == input.rank() - M + R_idx - 1);
  int32_t s_coord_dim = -1;
  if (has_coordinate_dim) {
    if (slinky::is_constant(index.extents[R_idx - 1], M)) {
      s_coord_dim = static_cast<int32_t>(R_idx) - 1;
    } else {
      for (size_t d = 0; d < R_idx - 1; ++d) {
        if (slinky::is_constant(index.extents[d], M)) {
          s_coord_dim = d;
          break;
        }
      }
    }
    if (s_coord_dim == -1) {
      s_coord_dim = static_cast<int32_t>(R_idx) - 1;
    }
  }
  size_t lookup_rank = has_coordinate_dim ? R_idx - 1 : R_idx;

  // Infer output shape.
  output.extents.resize(output_rank);
  auto get_index_dim = [has_coordinate_dim, s_coord_dim](size_t i) {
    if (has_coordinate_dim && i >= static_cast<size_t>(s_coord_dim)) {
      return i + 1;
    }
    return i;
  };

  bool is_aligned =
      (lookup_rank == output_rank) && (input.rank() == output_rank);
  if (is_aligned) {
    for (size_t d = 0; d < output_rank; ++d) {
      size_t s_idx_dim = get_index_dim(d);
      bool index_is_broadcast = slinky::is_one(index.extents[s_idx_dim]);

      if (index_is_broadcast) {
        subgraph.infer_elementwise_shape(node, /*input_idx=*/0,
                                         /*output_idx=*/0,
                                         /*input_dim=*/d,
                                         /*output_dim=*/d);
      } else {
        subgraph.infer_elementwise_shape(node, /*input_idx=*/1,
                                         /*output_idx=*/0,
                                         /*input_dim=*/s_idx_dim,
                                         /*output_dim=*/d);

        bool is_gathered =
            std::find(axes.begin(), axes.end(), d) != axes.end();
        if (!is_gathered) {
          bool input_is_one = slinky::is_one(input.extents[d]);
          if (!input_is_one) {
            subgraph.infer_elementwise_shape(node, /*input_idx=*/0,
                                             /*output_idx=*/0,
                                             /*input_dim=*/d,
                                             /*output_dim=*/d);
          }
        }
      }
    }
  } else {
    GatherMapping mapping =
        compute_gather_mapping(input.rank(), output_rank, lookup_rank, axes);

    for (size_t d = 0; d < output_rank; ++d) {
      int input_dim = mapping.output_to_input(d);
      if (input_dim != -1) {
        subgraph.infer_elementwise_shape(node, /*input_idx=*/0,
                                         /*output_idx=*/0,
                                         /*input_dim=*/input_dim,
                                         /*output_dim=*/d);
      } else {
        size_t i = d - mapping.s_out_start;
        size_t s_idx_dim = get_index_dim(i);
        subgraph.infer_elementwise_shape(node, /*input_idx=*/1,
                                         /*output_idx=*/0,
                                         /*input_dim=*/s_idx_dim,
                                         /*output_dim=*/d);
      }
    }
  }

  node.op = ynn_node::gather{std::move(axes), s_coord_dim};

  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const auto& gather_op = std::get<ynn_node::gather>(node.op);
    const auto& axes = gather_op.axes;
    int32_t s_coord_dim = gather_op.s_coord_dim;
    bool has_coordinate_dim = (s_coord_dim != -1);
    ynn_runtime_value& input = runtime.value(node.inputs[0]);
    ynn_runtime_value& index = runtime.value(node.inputs[1]);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);
    const size_t output_rank = output.rank();
    // M is the number of dimensions in the input tensor that are being
    // gathered. For GatherNd, this is the size of the coordinate
    // dimension (last dim of index).
    size_t M = axes.size();
    // R_idx is the rank of the index tensor.
    size_t R_idx = index.rank();
    size_t lookup_rank = has_coordinate_dim ? R_idx - 1 : R_idx;

    output.make_buffer(runtime, input.buffer->elem_size());

    require_contiguous(*input.buffer, 1);
    require_contiguous(*index.buffer, 1);
    require_contiguous(*output.buffer, 1);

    std::vector<slinky::var> dims = runtime.globals.make_dims(output_rank);

    bool is_aligned =
        (lookup_rank == output_rank) && (input.rank() == output_rank);
    slinky::box_expr input_bounds(input.rank());
    slinky::box_expr index_bounds(index.rank());
    auto get_index_dim = [has_coordinate_dim, s_coord_dim](size_t i) {
      if (has_coordinate_dim && i >= static_cast<size_t>(s_coord_dim)) {
        return i + 1;
      }
      return i;
    };

    if (is_aligned) {
      for (size_t d = 0; d < output_rank; ++d) {
        if (std::find(axes.begin(), axes.end(), d) == axes.end()) {
          size_t s_idx_dim = get_index_dim(d);
          bool input_is_dummy =
              slinky::is_one(input.physical_extents()[d]) &&
              !slinky::is_one(index.physical_extents()[s_idx_dim]);
          if (!input_is_dummy) {
            input_bounds[d] =
                elementwise_bounds(dims[d], input.physical_extents()[d]);
          }
        }
      }
      for (size_t d : axes) {
        if (d < input_bounds.size()) {
          input_bounds[d] = all_bounds(input.physical_extents()[d]);
        }
      }
      for (size_t d = 0; d < lookup_rank; ++d) {
        size_t s_idx_dim = get_index_dim(d);
        index_bounds[s_idx_dim] =
            elementwise_bounds(dims[d], index.physical_extents()[s_idx_dim]);
      }
    } else {
      GatherMapping mapping =
          compute_gather_mapping(input.rank(), output_rank, lookup_rank, axes);

      for (size_t d = 0; d < output_rank; ++d) {
        int input_dim = mapping.output_to_input(d);
        if (input_dim != -1) {
          input_bounds[input_dim] = elementwise_bounds(
              dims[d], input.physical_extents()[input_dim]);
        }
      }
      for (size_t d : axes) {
        if (d < input_bounds.size()) {
          input_bounds[d] = all_bounds(input.physical_extents()[d]);
        }
      }


      for (size_t i = 0; i < lookup_rank; ++i) {
        size_t s_idx_dim = get_index_dim(i);
        size_t s_out = mapping.s_out_start + i;
        index_bounds[s_idx_dim] = elementwise_bounds(
            dims[s_out], index.physical_extents()[s_idx_dim]);
      }
    }

    if (has_coordinate_dim) {
      index_bounds[s_coord_dim] = all_bounds(M);
    }
    size_t index_elem_count = type_element_count(index.type);
    if (index_elem_count != 1 && !index_bounds.empty()) {
      index_bounds[0] /= (int)index_elem_count;
    }

    bool can_use_lut = axes.size() == 1 && axes[0] == 0;
    lut_kernel_fn kernel =
        can_use_lut ? get_lut_kernel(index.type, type_size_bits(input.type))
                    : nullptr;

    slinky::func func;
    if (kernel) {
      slinky::call_stmt::attributes attrs;
      attrs.name = "lut";
      func = slinky::func::make(make_lut_impl(kernel),
                                {{input.buffer, std::move(input_bounds)},
                                 {index.buffer, std::move(index_bounds)}},
                                {{output.buffer, dims}}, std::move(attrs));
    } else {
      slinky::call_stmt::attributes attrs;
      attrs.name = "gather";
      std::vector<bool> input_dims_collapsible(input.rank());
      for (size_t d = 0; d < input.rank(); ++d) {
        input_dims_collapsible[d] = slinky::is_one(input.physical_extents()[d]);
      }
      std::vector<bool> index_dims_collapsible;
      for (size_t d = 0; d < index.rank(); ++d) {
        if (has_coordinate_dim && d == static_cast<size_t>(s_coord_dim)) {
          continue;
        }
        index_dims_collapsible.push_back(
            slinky::is_one(index.physical_extents()[d]));
      }
      func = slinky::func::make(
          make_gather_impl(axes, input.rank(), output_rank, index.rank(),
                           index.type, s_coord_dim,
                           std::move(input_dims_collapsible),
                           std::move(index_dims_collapsible)),
          {{input.buffer, std::move(input_bounds)},
           {index.buffer, std::move(index_bounds)}},
          {{output.buffer, dims}}, std::move(attrs));
    }

    auto sched = runtime.make_schedule(dims, output.physical_extents(),
                                       output.buffer->elem_size());
    func.user_data() = sched.get();
    runtime.scheduling_info_storage.push_back(std::move(sched));
    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };
}

}  // namespace ynn

extern "C" {

ynn_status ynn_define_gather(ynn_subgraph_t subgraph, size_t num_axes,
                             const int32_t* axes, size_t output_rank,
                             uint32_t input_id, uint32_t index_id,
                             uint32_t* output_id, uint32_t flags) {
  YNN_RETURN_IF_ERROR(ynn::validate_subgraph("gather", subgraph));
  YNN_RETURN_IF_ERROR(
      ynn::validate_input_tensor("gather", subgraph, "input_id", input_id));
  YNN_RETURN_IF_ERROR(
      ynn::validate_input_tensor("gather", subgraph, "index_id", index_id));
  YNN_RETURN_IF_ERROR(
      ynn::validate_output_tensor("gather", subgraph, "output_id", output_id));

  if (num_axes == 0) {
    YNN_LOG_ERROR() << "For node `gather`, num_axes must be greater than 0";
    return ynn_status_invalid_parameter;
  }

  const ynn_value& input = subgraph->value(input_id);
  const ynn_value& index = subgraph->value(index_id);

  if (!ynn::type_is_integral(index.type)) {
    YNN_LOG_ERROR() << "For node `gather`, index must be integral, got "
                    << index.type;
    return ynn_status_invalid_parameter;
  }

  if (*output_id != YNN_INVALID_VALUE_ID) {
    const ynn_value& output = subgraph->value(*output_id);
    if (ynn::type_size_bits(output.type) != ynn::type_size_bits(input.type)) {
      YNN_LOG_ERROR()
          << "For node `gather`, input and output types must be the "
             "same size, got "
          << input.type << " and " << output.type;
      return ynn_status_invalid_parameter;
    }
  }

  std::vector<int32_t> axes_vec(num_axes);
  for (size_t i = 0; i < num_axes; ++i) {
    YNN_RETURN_IF_ERROR(
        ynn::validate_axis("gather", "input", input.rank(), axes[i]));
    axes_vec[i] = ynn::axis_to_slinky_dim(input.rank(), axes[i]);
  }

  ynn_node node;
  ynn::define_gather(*subgraph, node, std::move(axes_vec), output_rank,
                     input_id, index_id, *output_id);
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

}  // extern "C"
