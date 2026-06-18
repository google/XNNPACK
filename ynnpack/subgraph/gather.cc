// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
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
  return [kernel](slinky::raw_buffer a, slinky::raw_buffer lut,
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
    lut.slice(0, 0);

    slinky::for_each_element(
        [=](void* x, const void* a, const void* lut_ptr) {
          kernel(x_n_extent, a, lut_ptr, x);
        },
        x, a, lut);
    return 0;
  };
}

ynn_status create_gather_lut(const ynn_node& node, ynn_runtime& runtime,
                             lut_kernel_fn kernel) {
  assert(node.inputs.size() == 2);
  assert(node.outputs.size() == 1);

  const ynn_runtime_value& lut = runtime.value(node.inputs[0]);  // table
  const ynn_runtime_value& a = runtime.value(node.inputs[1]);    // index
  ynn_runtime_value& x = runtime.value(node.outputs[0]);

  x.make_buffer(runtime);
  std::vector<slinky::var> dims = runtime.globals.make_dims(x.rank());
  slinky::box_expr bounds = make_elementwise_bounds(dims, a.physical_extents());

  slinky::box_expr lut_bounds(lut.rank());
  lut_bounds[0] = slinky::interval_expr(0, 1 << type_size_bytes(a.type));
  for (size_t d = 1; d < lut.rank(); ++d) {
    lut_bounds[d] = elementwise_bounds(dims[d], lut.physical_extents()[d]);
  }

  slinky::call_stmt::attributes attrs;
  attrs.name = "lut";
  attrs.allow_in_place = compute_allow_in_place(node, *runtime.subgraph);

  auto func = slinky::func::make(
      make_lut_impl(kernel),
      {{a.buffer, std::move(bounds)}, {lut.buffer, std::move(lut_bounds)}},
      {{x.buffer, dims}}, std::move(attrs));

  auto sched =
      runtime.make_schedule(dims, x.physical_extents(), x.buffer->elem_size());
  func.user_data() = sched.get();
  runtime.scheduling_info_storage.push_back(std::move(sched));
  runtime.funcs.push_back(std::move(func));

  return ynn_status_success;
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

auto make_gather_impl(int32_t axis, ynn_type index_type) {
  return
      [axis, index_type](
          slinky::buffer<const void, YNN_MAX_TENSOR_RANK> input,
          slinky::buffer<const void, YNN_MAX_TENSOR_RANK> index,
          slinky::buffer<void, YNN_MAX_TENSOR_RANK> output) -> slinky::index_t {
        slinky::dim input_axis = input.dim(axis);

        std::size_t elem_size = output.elem_size;
        const size_t R_loop = index.rank;
        const size_t R_rem = output.rank - R_loop;

        // 1. Modify input in place to crop the axis.
        if (index.rank == 0) {
          input.slice(axis, input_axis.min());
        } else {
          input.crop(axis, input_axis.min(), input_axis.min());
          if (axis < input.rank) {
            input.mutable_dim(axis).set_stride(0);
          }
        }

        // 3. Prepare slices for copy outside the loop.
        // They represent the outermost R_rem dimensions.
        slinky::raw_buffer output_slice;
        output_slice.elem_size = elem_size;
        output_slice.rank = R_rem;
        output_slice.dims = (R_rem > 0) ? (output.dims + R_loop) : nullptr;

        slinky::raw_buffer input_slice;
        input_slice.elem_size = elem_size;
        input_slice.rank = (input.rank > static_cast<size_t>(R_loop))
                               ? (input.rank - static_cast<size_t>(R_loop))
                               : 0;
        input_slice.dims = (input.rank > static_cast<size_t>(R_loop))
                               ? (input.dims + R_loop)
                               : nullptr;

        // Buffers for the for_each_element loop.
        // They represent the innermost R_loop dimensions.
        slinky::raw_buffer output_for_loop = output;
        output_for_loop.dims = output.dims;
        output_for_loop.rank = R_loop;

        slinky::raw_buffer input_for_loop = input;
        input_for_loop.dims = input.dims;
        input_for_loop.rank =
            std::min<size_t>(input.rank, static_cast<size_t>(R_loop));

        // 4. Run gather.
        bool error = false;
        slinky::for_each_element(
            [=, &error, &output_slice, &input_slice](
                void* output_ptr, const void* index_ptr,
                const void* input_dummy_ptr) {
              slinky::index_t idx = read_index_value(index_ptr, index_type);

              if (input_axis.contains(idx)) {
                const void* input_ptr = slinky::offset_bytes(
                    input_dummy_ptr, input_axis.flat_offset_bytes(idx));

                output_slice.base = output_ptr;
                input_slice.base = const_cast<void*>(input_ptr);

                slinky::copy(input_slice, output_slice);
              } else {
                error = true;
              }
            },
            output_for_loop, index, input_for_loop);

        return error;
      };
}

}  // namespace

void define_gather(ynn_subgraph& subgraph, ynn_node& node, int32_t axis,
                   uint32_t input_id, uint32_t index_id, uint32_t& output_id) {
  const ynn_value& input = subgraph.value(input_id);
  const ynn_value& index = subgraph.value(index_id);

  ynn_value& output = subgraph.get_output_value(&output_id, input);

  node.inputs = {input_id, index_id};
  node.outputs = {output_id};
  node.op = ynn_node::gather{axis};

  size_t output_rank = index.rank();
  for (size_t d = 0; d < input.rank(); ++d) {
    if (d != static_cast<size_t>(axis)) {
      output_rank = std::max(output_rank, d + 1);
    }
  }
  output.extents.resize(output_rank);

  for (size_t d = 0; d < index.rank(); ++d) {
    subgraph.infer_elementwise_shape(node, /*input_idx=*/1,
                                     /*output_idx=*/0,
                                     /*input_dim=*/d, /*output_dim=*/d);
  }

  for (size_t d = 0; d < input.rank(); ++d) {
    if (d != static_cast<size_t>(axis)) {
      subgraph.infer_elementwise_shape(node, /*input_idx=*/0,
                                       /*output_idx=*/0,
                                       /*input_dim=*/d, /*output_dim=*/d);
    }
  }

  if (axis == 0 && type_size_bits(index.type) <= 16) {
    // If we are doing the gather in axis 0, we might be able to use a LUT
    // kernel for this.
    int expected_size = 1 << type_size_bits(index.type);
    if (!slinky::is_constant(input.extent(0), expected_size)) {
      // We can't use a LUT kernel if the index might be out of bounds.
    } else {
      lut_kernel_fn kernel = get_lut_kernel(index.type, input.type);
      if (kernel) {
        node.create = [kernel](const ynn_node& node, ynn_runtime& runtime) {
          return create_gather_lut(node, runtime, kernel);
        };
        return;
      }
    }
  }

  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    int32_t axis = std::get<ynn_node::gather>(node.op).axis;
    const ynn_runtime_value& input = runtime.value(node.inputs[0]);
    const ynn_runtime_value& index = runtime.value(node.inputs[1]);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    output.make_buffer(runtime, input.buffer->elem_size());

    std::vector<slinky::var> dims = runtime.globals.make_dims(output.rank());

    slinky::box_expr input_bounds(input.rank());
    for (size_t d = 0; d < input.rank(); ++d) {
      if (d == static_cast<size_t>(axis)) {
        input_bounds[d] = all_bounds(input.physical_extents()[d]);
      } else {
        input_bounds[d] =
            elementwise_bounds(dims[d], input.physical_extents()[d]);
      }
    }

    slinky::box_expr index_bounds(index.rank());
    for (size_t j = 0; j < index.rank(); ++j) {
      index_bounds[j] =
          elementwise_bounds(dims[j], index.physical_extents()[j]);
    }

    auto func = slinky::func::make(make_gather_impl(axis, index.type),
                                   {{input.buffer, std::move(input_bounds)},
                                    {index.buffer, std::move(index_bounds)}},
                                   {{output.buffer, dims}});

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

ynn_status ynn_define_gather(ynn_subgraph_t subgraph, int32_t axis,
                             uint32_t input_id, uint32_t index_id,
                             uint32_t* output_id, uint32_t flags) {
  YNN_RETURN_IF_ERROR(ynn::validate_subgraph("gather", subgraph));
  YNN_RETURN_IF_ERROR(
      ynn::validate_input_tensor("gather", subgraph, "input_id", input_id));
  YNN_RETURN_IF_ERROR(
      ynn::validate_input_tensor("gather", subgraph, "index_id", index_id));
  YNN_RETURN_IF_ERROR(
      ynn::validate_output_tensor("gather", subgraph, "output_id", output_id));
  const ynn_value& input = subgraph->value(input_id);
  YNN_RETURN_IF_ERROR(
      ynn::validate_axis("gather", "input", input.rank(), axis));
  const ynn_value& index = subgraph->value(index_id);
  if (input.rank() > index.rank() &&
      !(index.rank() == 0 && input.rank() == 1)) {
    YNN_LOG_ERROR()
        << "For node `gather`, input rank must be less than or equal to index "
           "rank (unless index is scalar and input is 1D). Got input rank "
        << input.rank() << " and index rank " << index.rank();
    return ynn_status_invalid_parameter;
  }
  if (!ynn::type_is_integral(index.type)) {
    YNN_LOG_ERROR() << "For node `gather`, index must be integral, got "
                    << index.type;
    return ynn_status_invalid_parameter;
  }

  axis = ynn::axis_to_slinky_dim(input.rank(), axis);

  ynn_node node;
  ynn::define_gather(*subgraph, node, axis, input_id, index_id, *output_id);
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

}  // extern "C"
