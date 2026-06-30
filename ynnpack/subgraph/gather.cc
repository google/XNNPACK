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
#include <utility>
#include <vector>

#include "ynnpack/base/log.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/lut/lut.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/utils.h"
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

auto make_gather_impl(std::vector<int32_t> gathered_axes, size_t output_rank,
                      ynn_type index_type) {
  return
      [gathered_axes = std::move(gathered_axes), output_rank, index_type](
          slinky::buffer<const void, YNN_MAX_TENSOR_RANK> input,
          slinky::buffer<const void, YNN_MAX_TENSOR_RANK> index,
          slinky::buffer<void, YNN_MAX_TENSOR_RANK> output) -> slinky::index_t {
        slinky::dim axis_index_dim = index.dim(output_rank);
        index.slice(output_rank);

        // We're going to address the gathered dimensions separately from the
        // loop over the output.
        slinky::dim gathered_input_dims[YNN_MAX_TENSOR_RANK];
        size_t num_gathered_dims = gathered_axes.size();
        for (size_t i = 0; i < num_gathered_dims; ++i) {
          gathered_input_dims[i] = input.dim(gathered_axes[i]);
          input.mutable_dim(gathered_axes[i]).set_stride(0);
        }

        // We need two sets of buffers: one for defining an outer loop, and one
        // to define how we can slinky::copy inside that outer loop.
        slinky::buffer<void, YNN_MAX_TENSOR_RANK> output_slice = output;
        slinky::buffer<const void, YNN_MAX_TENSOR_RANK> input_slice = input;

        for (int i = output.rank - 1; i >= 0; --i) {
          if (index.dim(i).is_broadcast()) {
            // The index is a broadcast, we should handle it with slinky::copy.
            output.slice(i);
            index.slice(i);
            input.slice(i);
          } else {
            // The index is not a broadcast, we should handle it with the outer
            // loop.
            output_slice.slice(i);
            input_slice.slice(i);
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

  ynn_value& output = subgraph.get_output_value(&output_id, input);

  node.inputs = {input_id, index_id};
  node.outputs = {output_id};

  // Infer output shape.
  output.extents.resize(output_rank);
  for (size_t d = 0; d < output_rank; ++d) {
    subgraph.infer_elementwise_shape(node, /*input_idx=*/1,
                                     /*output_idx=*/0,
                                     /*input_dim=*/d,
                                     /*output_dim=*/d);

    if (std::find(axes.begin(), axes.end(), d) == axes.end()) {
      // This dimension is not gathered.
      subgraph.infer_elementwise_shape(node, /*input_idx=*/0,
                                       /*output_idx=*/0,
                                       /*input_dim=*/d,
                                       /*output_dim=*/d);
    }
  }

  node.op = ynn_node::gather{std::move(axes)};

  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const auto& axes = std::get<ynn_node::gather>(node.op).axes;
    ynn_runtime_value& input = runtime.value(node.inputs[0]);
    ynn_runtime_value& index = runtime.value(node.inputs[1]);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);
    const size_t output_rank = output.rank();

    output.make_buffer(runtime, input.buffer->elem_size());

    require_contiguous(*input.buffer, 1);
    require_contiguous(*index.buffer, 1);
    require_contiguous(*output.buffer, 1);

    std::vector<slinky::var> dims = runtime.globals.make_dims(output_rank);

    slinky::box_expr input_bounds =
        make_elementwise_bounds(dims, input.physical_extents());
    for (size_t d : axes) {
      // We need all of the gathered dimensions.
      if (d < input_bounds.size()) {
        input_bounds[d] = all_bounds(input.physical_extents()[d]);
      }
    }

    slinky::box_expr index_bounds =
        make_elementwise_bounds(dims, index.physical_extents());
    size_t index_elem_count = type_element_count(index.type);
    if (index_elem_count != 1 && !index_bounds.empty()) {
      index_bounds[0] /= (int)index_elem_count;
    }
    if (index.rank() > output_rank) {
      // We need an index for each axis we are gathering.
      index_bounds.push_back(all_bounds(axes.size()));
    }

    bool can_use_lut = axes.size() == 1 && axes[0] == 0;
    lut_kernel_fn kernel =
        can_use_lut ? get_lut_kernel(index.type, type_size_bits(input.type))
                    : nullptr;

    slinky::func func;
    if (kernel) {
      slinky::call_stmt::attributes attrs;
      attrs.name = "lut";
      attrs.allow_in_place = compute_allow_in_place(node, *runtime.subgraph);
      func = slinky::func::make(make_lut_impl(kernel),
                                {{input.buffer, std::move(input_bounds)},
                                 {index.buffer, std::move(index_bounds)}},
                                {{output.buffer, dims}}, std::move(attrs));
    } else {
      slinky::call_stmt::attributes attrs;
      attrs.name = "gather";
      attrs.allow_in_place = compute_allow_in_place(node, *runtime.subgraph);
      func = slinky::func::make(make_gather_impl(axes, output_rank, index.type),
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
