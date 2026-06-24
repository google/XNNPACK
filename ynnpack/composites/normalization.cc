// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>

#include "ynnpack/composites/composites.h"
#include "ynnpack/composites/util.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

ynn_status define_softmax(ynn_subgraph_t subgraph, uint32_t input_id,
                          float beta, uint32_t& output_id) {
  uint32_t scaled_input_id = input_id;
  if (beta != 1.0f) {
    // Multiply input by beta.
    uint32_t beta_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(define_constant(subgraph, beta, beta_id));
    uint32_t multiply_output_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply,
                                          input_id, beta_id,
                                          &multiply_output_id, 0));
    scaled_input_id = multiply_output_id;
  }

  // 1. Max reduction along last axis.
  uint32_t max_input_id = YNN_INVALID_VALUE_ID;
  const int32_t last_axis[] = {-1};
  YNN_RETURN_IF_ERROR(ynn_define_reduce(
      subgraph, ynn_reduce_max, 1, last_axis, scaled_input_id,
      YNN_INVALID_VALUE_ID, &max_input_id, YNN_NODE_FLAG_KEEP_DIMS));

  // 2. Subtract max from scaled input.
  uint32_t input_minus_max_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_subtract,
                                        scaled_input_id, max_input_id,
                                        &input_minus_max_id, 0));

  // 3. Exp.
  uint32_t exp_input_minus_max_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_unary(
      subgraph, ynn_unary_exp, input_minus_max_id, &exp_input_minus_max_id, 0));

  // 4. Sum reduction of exp along last axis.
  uint32_t sum_exp_input_minus_max_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(
      ynn_define_reduce(subgraph, ynn_reduce_sum, 1, last_axis,
                        exp_input_minus_max_id, YNN_INVALID_VALUE_ID,
                        &sum_exp_input_minus_max_id, YNN_NODE_FLAG_KEEP_DIMS));

  // 5. Reciprocal of sum (1.0 / sum).
  uint32_t one_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(define_constant(subgraph, 1.0f, one_id));

  uint32_t inv_sum_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_divide, one_id,
                                        sum_exp_input_minus_max_id, &inv_sum_id,
                                        0));

  // 6. Multiply exp by reciprocal of sum.
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply,
                                        exp_input_minus_max_id, inv_sum_id,
                                        &output_id, 0));

  return ynn_status_success;
}

ynn_status define_log_softmax(ynn_subgraph_t subgraph, uint32_t input_id,
                              uint32_t& output_id) {
  // LogSoftmax(x)_i = x_i - max(x) - log(sum(exp(x_j - max(x))))

  // 1. Max reduction along last axis.
  uint32_t max_input_id = YNN_INVALID_VALUE_ID;
  const int32_t last_axis[] = {-1};
  YNN_RETURN_IF_ERROR(ynn_define_reduce(
      subgraph, ynn_reduce_max, 1, last_axis, input_id, YNN_INVALID_VALUE_ID,
      &max_input_id, YNN_NODE_FLAG_KEEP_DIMS));

  // 2. Subtract max from input.
  uint32_t input_minus_max_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_subtract, input_id,
                                        max_input_id, &input_minus_max_id, 0));

  // 3. Exp.
  uint32_t exp_input_minus_max_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_unary(
      subgraph, ynn_unary_exp, input_minus_max_id, &exp_input_minus_max_id, 0));

  // 4. Sum reduction of exp along last axis.
  uint32_t sum_exp_input_minus_max_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(
      ynn_define_reduce(subgraph, ynn_reduce_sum, 1, last_axis,
                        exp_input_minus_max_id, YNN_INVALID_VALUE_ID,
                        &sum_exp_input_minus_max_id, YNN_NODE_FLAG_KEEP_DIMS));

  // 5. Log of sum.
  uint32_t log_sum_exp_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_unary(
      subgraph, ynn_unary_log, sum_exp_input_minus_max_id, &log_sum_exp_id, 0));

  // 6. Subtract log_sum_exp from (input - max).
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_subtract,
                                        input_minus_max_id, log_sum_exp_id,
                                        &output_id, 0));

  return ynn_status_success;
}

}  // namespace ynn
