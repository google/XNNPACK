// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "ynnpack/base/base.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/log.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/dot/dot.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/xnnpack/utils.h"
#include "slinky/builder/pipeline.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

namespace ynn {

using slinky::index_t;

namespace {

// Returns true if we should rewrite int8 dynamic quantization to uint8, because
// we have faster dot kernels for uint8 x int8 than we do int8 x int8.
bool should_use_uint8() {
  // Get the kernels we would use for both int8 and uint8.
  dot_kernel int8 =
      get_dot_kernel({ynn_type_int8, ynn_type_int8, ynn_type_int32});
  dot_kernel uint8 =
      get_dot_kernel({ynn_type_uint8, ynn_type_int8, ynn_type_int32});
  if (!uint8.kernel) {
    return false;
  }

  // Find a size that is a common multiple of both kernels.
  const size_t m = int8.block_m * uint8.block_m;
  const size_t n = int8.block_n * uint8.block_n;
  const size_t k = int8.block_k * uint8.block_k;
  constexpr size_t tile_m = 1;

  // Estimate the cost of both kernels.
  const float uint8_cost =
      estimate_dot_cost(m, n, k, uint8.block_m, uint8.block_n, uint8.block_k,
                        tile_m, uint8.tile_n, uint8.tile_k);
  const float int8_cost =
      estimate_dot_cost(m, n, k, int8.block_m, int8.block_n, int8.block_k,
                        tile_m, int8.tile_n, int8.tile_k);
  return uint8_cost < int8_cost;
}

// This is a near-duplicate of that found in src/xnnpack/quantization.h
template <typename T>
std::pair<float, int32_t> compute_qd8_params(T min, T max) {
  constexpr float qmin = std::numeric_limits<uint8_t>::min();
  constexpr float qmax = std::numeric_limits<uint8_t>::max();
  const float rmin = std::min<float>(0.0f, min);
  const float rmax = std::max<float>(0.0f, max);
  const float scale = rmin == rmax ? 1.f : (qmax - qmin) / (rmax - rmin);
  const float descaled_min = rmin * scale;
  const float descaled_max = rmax * scale;
  const float zero_point_from_min_error = qmin + descaled_min;
  const float zero_point_from_max_error = qmax + descaled_max;
  float zero_point = zero_point_from_min_error + zero_point_from_max_error > 0
                         ? qmin - descaled_min
                         : qmax - descaled_max;
  zero_point = std::max<float>(zero_point, qmin);
  zero_point = std::min<float>(zero_point, qmax);
  assert(zero_point >= std::numeric_limits<uint8_t>::min());
  assert(zero_point <= std::numeric_limits<uint8_t>::max());
  // We compute the zero point for uint8, but we want the result to be for int8.
  // This means we can assume the (float) zero point is positive.
  const int32_t nudged_zero_point =
      static_cast<int32_t>(zero_point + 0.5f) - 128;
  return {1.0f / scale, nudged_zero_point};
}

// Call compute_qd8_params for each element.
template <typename T>
auto make_compute_qd8_params_impl(int32_t output_zero_point) {
  return [=](const slinky::buffer<const T>& min_max,
             const slinky::buffer<float>& scale,
             const slinky::buffer<int32_t>& zero_point) -> index_t {
    const index_t index_stride_bytes =
        min_max.dim(min_max.rank - 1).stride();
    assert(index_stride_bytes % sizeof(T) == 0);
    const index_t index_stride = index_stride_bytes / sizeof(T);
    slinky::for_each_contiguous_slice(scale,
        [=](index_t n, float* scale, int32_t* zero_point, const T* min_max) {
          for (index_t i = 0; i < n; ++i) {
            std::tie(scale[i], zero_point[i]) =
                compute_qd8_params(min_max[i], min_max[i + index_stride]);
            zero_point[i] += output_zero_point;
          }
        },
        zero_point, min_max);
    return 0;
  };
}

ynn_status define_qd8_params(ynn_subgraph_t subgraph, size_t num_nonbatch_axes,
                             const int32_t* nonbatch_axes,
                             int32_t output_zero_point, uint32_t min_max_id,
                             uint32_t scale_id, uint32_t zero_point_id) {
  // Validate arguments.
  assert(subgraph);
  assert(subgraph->is_valid_value(min_max_id));
  assert(subgraph->is_valid_value(scale_id));
  assert(subgraph->is_valid_value(zero_point_id));
  const ynn_value& min_max_value = subgraph->value(min_max_id);
  ynn_value& scale = subgraph->value(scale_id);
  ynn_value& zero_point = subgraph->value(zero_point_id);
  assert(scale.type == ynn_type_fp32);
  assert(zero_point.type == ynn_type_int32);

  // Propagate shape.
  // We've already done the reductions, so this is elementwise.
  scale.extents = min_max_value.extents;
  scale.extents.pop_back();
  zero_point.extents = scale.extents;

  // Make the node.
  ynn_node node;
  node.inputs = {min_max_id};
  node.outputs = {scale_id, zero_point_id};
  node.op = ynn_node::opaque{"compute_qd8_params"};
  node.create = [output_zero_point](const ynn_node& node,
                                    ynn_runtime& runtime) {
    const ynn_runtime_value& min_max = runtime.value(node.inputs[0]);
    ynn_runtime_value& scale = runtime.value(node.outputs[0]);
    ynn_runtime_value& zero_point = runtime.value(node.outputs[1]);

    scale.make_buffer(runtime);
    zero_point.make_buffer(runtime);

    std::vector<slinky::var> dims = make_dims(scale.rank(), runtime.symbols);
    slinky::box_expr min_max_bounds =
        make_elementwise_bounds(dims, min_max.extents);
    min_max_bounds.push_back(slinky::min_extent(0, 2));

    slinky::func func;
    slinky::call_stmt::attributes attrs;
    attrs.name = "compute_qd8_params";
    switch (min_max.type) {
      case ynn_type_fp32:
        func = slinky::func::make(
            make_compute_qd8_params_impl<float>(output_zero_point),
            {{min_max.buffer, std::move(min_max_bounds)}},
            {{scale.buffer, dims}, {zero_point.buffer, dims}}, attrs);
        break;
      case ynn_type_fp16:
        func = slinky::func::make(
            make_compute_qd8_params_impl<half>(output_zero_point),
            {{min_max.buffer, std::move(min_max_bounds)}},
            {{scale.buffer, dims}, {zero_point.buffer, dims}}, attrs);
        break;
      default:
        YNN_UNREACHABLE;
    }

    auto sched = runtime.make_schedule(dims, scale.buffer, node.outputs[0]);

    // `make_schedule` schedules the scale output buffer, but we
    // also need to schedule the zero point buffer too.
    // TODO(dsharlet): We should have a way to handle these buffers
    // consistently.
    ynn::scheduled_buffer sched_output_buffer = {zero_point.buffer, 0};
    sched->scheduled_buffers.push_back(std::move(sched_output_buffer));

    func.user_data() = sched.get();
    runtime.scheduling_info_storage.push_back(std::move(sched));

    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

}  // namespace

ynn_status compute_qd8_params(ynn_subgraph_t subgraph, size_t num_nonbatch_axes,
                              uint32_t input_id, uint32_t output_id) {
  ynn_value& output = subgraph->value(output_id);

  const ynn_value& scale = subgraph->value(output.scale_id);
  const ynn_value& zero_point = subgraph->value(output.zero_point_id);
  assert(scale.rank() == zero_point.rank());
  (void)scale;
  (void)zero_point;

  // We need the min and max of the input for each point in the scale/zero point
  // values.
  uint32_t min_identity_id = YNN_INVALID_VALUE_ID;
  ynn_status status = define_scalar_value_like(
      subgraph, input_id, std::numeric_limits<float>::infinity(),
      &min_identity_id);
  if (status != ynn_status_success) {
    return status;
  }

  uint32_t max_identity_id = YNN_INVALID_VALUE_ID;
  status = define_scalar_value_like(subgraph, input_id,
                                    -std::numeric_limits<float>::infinity(),
                                    &max_identity_id);
  if (status != ynn_status_success) {
    return status;
  }

  // XNNPACK defines dynamic quantization as a number of "nonbatch_dims", which
  // is the rank of the quantization data, and is always the last dimensions.
  // We need to compute the reduction over these dimensions.
  int32_t nonbatch_axes[YNN_MAX_TENSOR_RANK];
  for (int32_t i = 0; i < num_nonbatch_axes; ++i) {
    nonbatch_axes[i] = -i - 1;
  }

  uint32_t min_max_id = YNN_INVALID_VALUE_ID;
  status = ynn_define_reduce(subgraph, ynn_reduce_min_max, num_nonbatch_axes,
                             nonbatch_axes, input_id, YNN_INVALID_VALUE_ID,
                             &min_max_id,
                             /*flags=*/YNN_NODE_FLAG_KEEP_DIMS);
  if (status != ynn_status_success) {
    return status;
  }

  int32_t output_zero_point = 0;
  if (output.type == ynn_type_int8 && should_use_uint8() &&
      !output.is_external_output()) {
    // We can do uint8 dots faster than int8, so convert this result to uint8.
    // TODO(dsharlet): This assumes the consumer of this dynamic quantization is
    // a dot. Is that always the case? I think so...
    YNN_LOG_DEBUG() << "Rewriting dynamic quantization of tensor " << output_id
                    << " to uint8";
    output.type = ynn_type_uint8;
    output_zero_point = 128;
  }

  status = define_qd8_params(subgraph, num_nonbatch_axes, nonbatch_axes,
                             output_zero_point, min_max_id, output.scale_id,
                             output.zero_point_id);
  if (status != ynn_status_success) {
    return status;
  }

  return ynn_status_success;
}

}  // namespace ynn
