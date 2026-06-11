// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/xnnpack/dynamic_quantization.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "ynnpack/base/log.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/dot/dot.h"
#include "ynnpack/subgraph/subgraph.h"

namespace ynn {

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

}  // namespace

ynn_status compute_qd8_params(ynn_subgraph_t subgraph, size_t num_nonbatch_axes,
                              uint32_t input_id, uint32_t output_id,
                              uint32_t scale_id, uint32_t zero_point_id) {
  ynn_value& output = subgraph->value(output_id);

  const ynn_value& scale = subgraph->value(scale_id);
  const ynn_value& zero_point = subgraph->value(zero_point_id);
  assert(scale.rank() == zero_point.rank());
  (void)scale;
  (void)zero_point;

  // XNNPACK defines dynamic quantization as a number of "nonbatch_dims", which
  // is the rank of the quantization data, and is always the last dimensions.
  // We need to compute the reduction over these dimensions.
  int32_t nonbatch_axes[YNN_MAX_TENSOR_RANK];
  for (int32_t i = 0; i < num_nonbatch_axes; ++i) {
    nonbatch_axes[i] = -i - 1;
  }

  uint32_t min_max_id = YNN_INVALID_VALUE_ID;
  ynn_status status = ynn_define_reduce(
      subgraph, ynn_reduce_min_max, num_nonbatch_axes, nonbatch_axes, input_id,
      YNN_INVALID_VALUE_ID, &min_max_id,
      /*flags=*/YNN_NODE_FLAG_KEEP_DIMS);
  if (status != ynn_status_success) {
    return status;
  }

  if (output.type == ynn_type_int8 && should_use_uint8() &&
      !output.is_external_output()) {
    // We can do uint8 dots faster than int8, so convert this result to uint8.
    // TODO(dsharlet): This assumes the consumer of this dynamic quantization is
    // a dot. Is that always the case? I think so...
    YNN_LOG_DEBUG() << "Rewriting dynamic quantization of tensor " << output_id
                    << " to uint8";
    output.type = ynn_type_uint8;
  }

  status = ynn_define_dynamic_quantization(subgraph, min_max_id, output.type,
                                           &zero_point_id, &scale_id,
                                           /*flags=*/0);
  if (status != ynn_status_success) {
    return status;
  }

  return ynn_status_success;
}

}  // namespace ynn
