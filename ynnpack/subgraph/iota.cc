// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/iota/iota.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "ynnpack/base/base.h"
#include "ynnpack/base/log.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/iota.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/builder/pipeline.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

namespace ynn {

namespace {

// It's a little tricky to implement this. We can't use
// slinky::for_each_element, because we need the index too, not just the
// pointers to the corresponding elements.
template <typename T>
void iota_impl(iota_kernel_fn kernel, T begin, const T* stride,
               slinky::raw_buffer output) {
  if (output.rank <= 1) {
    // This handles the rank 0 case, because output.dim(0) will be a broadcast
    // dimension of extent 1.
    const slinky::dim& dim_0 = output.dim(0);
    assert(is_contiguous(dim_0, sizeof(T)));
    assert(!dim_0.is_folded());
    begin = begin + dim_0.min() * stride[0];
    kernel(dim_0.extent(), &begin, stride, static_cast<T*>(output.base));
  } else {
    // Dimensions are backwards from the public API...
    const T stride_d = *stride++;
    const int d = output.rank - 1;
    const slinky::dim& dim_d = output.dim(d);
    for (slinky::index_t i = dim_d.min(); i <= dim_d.max(); ++i) {
      slinky::raw_buffer output_i = output;
      output_i.slice(d, i);
      iota_impl(kernel, static_cast<T>(begin + i * stride_d), stride, output_i);
    }
  }
}

template <typename T>
auto make_iota_impl(iota_kernel_fn kernel, const iota_params& params) {
  return
      [=](slinky::buffer<const T, 0> begin, slinky::buffer<const T, 1> stride,
          slinky::buffer<T, YNN_MAX_TENSOR_RANK> output) -> slinky::index_t {
        // Copy the params to the type of the iota.
        T scale = params.scale;
        T offset = params.offset;
        assert(scale == params.scale);
        assert(offset == params.offset);

        assert(is_contiguous(stride.dim(0), sizeof(T)));
        assert(!stride.dim(0).is_folded());
        assert(output.rank <= YNN_MAX_TENSOR_RANK);
        // Rather than try to handle broadcasting in iota_impl above, just
        // copy the strides here. We initialize it to 0, so we avoid possible
        // ubsan complaints when output.rank = 0.
        T stride_broadcast[YNN_MAX_TENSOR_RANK] = {0, };
        for (int i = 0; i < output.rank; ++i) {
          stride_broadcast[i] = stride(i) * scale;
        }
        iota_impl<T>(kernel, begin() * scale + offset,
                     stride_broadcast, output);
        return 0;
      };
}

}  // namespace

extern "C" {

ynn_status ynn_define_iota(ynn_subgraph_t subgraph, ynn_type type, size_t rank,
                           const size_t* dims, uint32_t begin_id,
                           uint32_t stride_id, uint32_t* output_id,
                           uint32_t flags) {
  // Validate arguments.
  YNN_RETURN_IF_ERROR(validate_subgraph("iota", subgraph));

  uint32_t scalar_zero_id = YNN_INVALID_VALUE_ID;
  if (begin_id == YNN_INVALID_VALUE_ID || stride_id == YNN_INVALID_VALUE_ID) {
    scalar_zero_id = subgraph->get_scalar_value_id(type, YNN_INVALID_VALUE_ID,
                                                   YNN_INVALID_VALUE_ID, 0.0f);
    if (begin_id == YNN_INVALID_VALUE_ID) begin_id = scalar_zero_id;
    if (stride_id == YNN_INVALID_VALUE_ID) stride_id = scalar_zero_id;
  }

  YNN_RETURN_IF_ERROR(
      validate_input_tensor("iota", subgraph, "begin_id", begin_id));
  YNN_RETURN_IF_ERROR(
      validate_input_tensor("iota", subgraph, "stride_id", stride_id));

  if (rank > 0) {
    const ynn_value& stride_val = subgraph->value(stride_id);
    if (stride_val.rank() > 1) {
      YNN_LOG_ERROR() << "For node 'iota', 'stride_id' must be a 1D tensor";
      return ynn_status_invalid_parameter;
    }
  }

  YNN_RETURN_IF_ERROR(
      validate_output_tensor("iota", subgraph, "output_id", output_id));

  if (*output_id == YNN_INVALID_VALUE_ID) {
    *output_id = subgraph->new_internal_value(type).id;
  }
  ynn_value& output = subgraph->value(*output_id);
  output.type = type;
  output.extents.resize(rank);
  for (size_t i = 0; i < rank; ++i) {
    output.extents[axis_to_slinky_dim(rank, i)] =
        static_cast<slinky::index_t>(dims[i]);
  }

  ynn_node node;
  node.inputs = {begin_id, stride_id};
  node.outputs = {*output_id};
  node.op = ynn_node::iota{};
  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const ynn_node::iota& op = std::get<ynn_node::iota>(node.op);
    const ynn_runtime_value& begin = runtime.value(node.inputs[0]);
    const ynn_runtime_value& stride = runtime.value(node.inputs[1]);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    output.make_buffer(runtime);

    iota_kernel_fn kernel = get_iota_kernel(output.type);
    if (!kernel) {
      YNN_LOG_ERROR() << "Unsupported type for iota: " << output.type;
      return ynn_status_unsupported_parameter;
    }

    std::vector<slinky::var> dims = runtime.globals.make_dims(output.rank());

    slinky::call_stmt::attributes attrs;
    attrs.name = "iota";
    slinky::box_expr stride_bounds = {all_bounds(output.rank())};

    auto sched = runtime.make_schedule(dims, output.physical_extents(),
                                       output.buffer->elem_size());

    if (output.type == ynn_type_int32) {
      runtime.funcs.push_back(slinky::func::make(
          make_iota_impl<int32_t>(kernel, op.params),
          {{begin.buffer, {}}, {stride.buffer, std::move(stride_bounds)}},
          {{output.buffer, std::move(dims)}}, std::move(attrs)));
    } else if (output.type == ynn_type_fp32) {
      runtime.funcs.push_back(slinky::func::make(
          make_iota_impl<float>(kernel, op.params),
          {{begin.buffer, {}}, {stride.buffer, std::move(stride_bounds)}},
          {{output.buffer, std::move(dims)}}, std::move(attrs)));
    } else {
      YNN_UNREACHABLE;
    }

    runtime.funcs.back().user_data() = sched.get();
    runtime.scheduling_info_storage.push_back(std::move(sched));

    return ynn_status_success;
  };
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

}  // extern "C"

}  // namespace ynn
