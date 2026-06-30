// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>

#include "ynnpack/composites/composites.h"
#include "ynnpack/composites/util.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

ynn_status define_reduce_sum(ynn_subgraph_t subgraph, size_t num_axes,
                             const int32_t* axes, uint32_t input_id,
                             uint32_t input_zero_point_id,
                             uint32_t input_scale_id, bool keep_dims, bool mean,
                             bool squared, ynn_type output_type,
                             uint32_t output_zero_point_id,
                             uint32_t output_scale_id, uint32_t& output_id) {
  if (input_zero_point_id != YNN_INVALID_VALUE_ID ||
      input_scale_id != YNN_INVALID_VALUE_ID ||
      output_zero_point_id != YNN_INVALID_VALUE_ID ||
      output_scale_id != YNN_INVALID_VALUE_ID) {
    if (squared) {
      return ynn_status_invalid_parameter;
    }
    // Quantized path.
    // Accumulate in int32.
    uint32_t accumulator_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(
        ynn_define_reduce(subgraph, ynn_reduce_sum, num_axes, axes, input_id,
                          YNN_INVALID_VALUE_ID, &accumulator_id,
                          keep_dims ? YNN_NODE_FLAG_KEEP_DIMS : 0));

    // Compute accumulator zero point: ZP_acc = N * ZP_in.
    uint32_t zp_acc_id = YNN_INVALID_VALUE_ID;
    if (input_zero_point_id != YNN_INVALID_VALUE_ID) {
      // Get N (reduction size) dynamically as int32.
      uint32_t N_int_id = YNN_INVALID_VALUE_ID;
      YNN_RETURN_IF_ERROR(ynn_define_get_tensor_shape(
          subgraph, num_axes, axes, ynn_type_int32, /*rank=*/0, input_id,
          &N_int_id, YNN_NODE_FLAG_RESHAPE_1D | YNN_NODE_FLAG_UNIQUE_DIMS));

      YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply,
                                            N_int_id, input_zero_point_id,
                                            &zp_acc_id, 0));
    }

    // Compute accumulator scale: Scale_acc = Scale_in / N.
    uint32_t scale_acc_id = YNN_INVALID_VALUE_ID;

    if (mean) {
      // Get N (reduction size) dynamically as fp32.
      uint32_t N_fp_id = YNN_INVALID_VALUE_ID;
      YNN_RETURN_IF_ERROR(ynn_define_get_tensor_shape(
          subgraph, num_axes, axes, ynn_type_fp32, /*rank=*/0, input_id,
          &N_fp_id, YNN_NODE_FLAG_RESHAPE_1D | YNN_NODE_FLAG_UNIQUE_DIMS));

      // Scale_acc = Scale_in / N.
      YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_divide,
                                            input_scale_id, N_fp_id,
                                            &scale_acc_id, 0));
    } else {
      // For sum, Scale_acc = Scale_in.
      scale_acc_id = input_scale_id;
    }

    // Dequantize accumulator (int32) to fp32.
    // TODO: b/527609911 - We should be able to just use `ynn_define_quantize`
    // with an int32 input, and we can do the scale/zero point math accordingly.
    uint32_t real_output_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(
        ynn_define_dequantize(subgraph, accumulator_id, zp_acc_id, scale_acc_id,
                              ynn_type_fp32, &real_output_id, 0));

    // Quantize to output.
    YNN_RETURN_IF_ERROR(ynn_define_quantize(subgraph, real_output_id,
                                            output_type, output_zero_point_id,
                                            output_scale_id, &output_id, 0));

    return ynn_status_success;
  } else {
    // Float path.
    ynn_reduce_operator reduce_op =
        squared ? ynn_reduce_sum_squared : ynn_reduce_sum;
    if (mean) {
      uint32_t sum_output_id = YNN_INVALID_VALUE_ID;
      YNN_RETURN_IF_ERROR(ynn_define_reduce(
          subgraph, reduce_op, num_axes, axes, input_id, YNN_INVALID_VALUE_ID,
          &sum_output_id, keep_dims ? YNN_NODE_FLAG_KEEP_DIMS : 0));

      uint32_t divisor_id = YNN_INVALID_VALUE_ID;
      YNN_RETURN_IF_ERROR(ynn_define_get_tensor_shape(
          subgraph, num_axes, axes, ynn_type_fp32, /*rank=*/0, input_id,
          &divisor_id, YNN_NODE_FLAG_RESHAPE_1D | YNN_NODE_FLAG_UNIQUE_DIMS));

      return ynn_define_binary(subgraph, ynn_binary_divide, sum_output_id,
                               divisor_id, &output_id, 0);
    } else {
      return ynn_define_reduce(subgraph, reduce_op, num_axes, axes, input_id,
                               YNN_INVALID_VALUE_ID, &output_id,
                               keep_dims ? YNN_NODE_FLAG_KEEP_DIMS : 0);
    }
  }
}

}  // namespace ynn
