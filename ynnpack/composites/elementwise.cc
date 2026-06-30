// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>
#include <cstdint>

#include "ynnpack/composites/composites.h"
#include "ynnpack/composites/util.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

ynn_status define_gelu(ynn_subgraph_t subgraph, uint32_t input_id,
                       uint32_t& output_id) {
  uint32_t x_sqrt2_over_2_val_id = YNN_INVALID_VALUE_ID;
  uint32_t coeff_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(
      define_constant(subgraph, std::sqrt(2.0f) / 2.0f, coeff_id));
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply, input_id,
                                        coeff_id, &x_sqrt2_over_2_val_id, 0));

  uint32_t erf_x_sqrt2_over_2_val_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_unary(subgraph, ynn_unary_erf,
                                       x_sqrt2_over_2_val_id,
                                       &erf_x_sqrt2_over_2_val_id, 0));

  uint32_t half_erf_val_id = YNN_INVALID_VALUE_ID;
  uint32_t half_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(define_constant(subgraph, 0.5f, half_id));
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply,
                                        erf_x_sqrt2_over_2_val_id, half_id,
                                        &half_erf_val_id, 0));

  uint32_t half_erf_plus_half_val_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_add,
                                        half_erf_val_id, half_id,
                                        &half_erf_plus_half_val_id, 0));

  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply, input_id,
                                        half_erf_plus_half_val_id, &output_id,
                                        0));
  return ynn_status_success;
}

ynn_status define_approx_gelu(ynn_subgraph_t subgraph, uint32_t input_id,
                              uint32_t& output_id) {
  const double sqrt_2_over_pi = std::sqrt(2 / M_PI);
  const double coefficients[] = {0.0, sqrt_2_over_pi, 0.0,
                                 sqrt_2_over_pi * 0.044715};

  uint32_t tanh_arg_val_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_unary_polynomial(
      subgraph, input_id, 3, coefficients, &tanh_arg_val_id, 0));

  uint32_t tanh_out_val_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_unary(subgraph, ynn_unary_tanh,
                                       tanh_arg_val_id, &tanh_out_val_id, 0));

  uint32_t one_plus_tanh_val_id = YNN_INVALID_VALUE_ID;
  uint32_t one_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(define_constant(subgraph, 1.0f, one_id));
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_add,
                                        tanh_out_val_id, one_id,
                                        &one_plus_tanh_val_id, 0));

  uint32_t x_times_half_val_id = YNN_INVALID_VALUE_ID;
  uint32_t half_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(define_constant(subgraph, 0.5f, half_id));
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply, input_id,
                                        half_id, &x_times_half_val_id, 0));

  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply,
                                        x_times_half_val_id,
                                        one_plus_tanh_val_id, &output_id, 0));
  return ynn_status_success;
}

ynn_status define_elu(ynn_subgraph_t subgraph, uint32_t input_id, float alpha,
                      uint32_t& output_id) {
  uint32_t min_x_0_val_id = YNN_INVALID_VALUE_ID;
  uint32_t zero_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(define_constant(subgraph, 0.0f, zero_id));
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_min, input_id,
                                        zero_id, &min_x_0_val_id, 0));

  uint32_t expm1_x_val_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_unary(subgraph, ynn_unary_expm1,
                                       min_x_0_val_id, &expm1_x_val_id, 0));

  uint32_t alpha_times_expm1_x_val_id = YNN_INVALID_VALUE_ID;
  uint32_t alpha_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(define_constant(subgraph, alpha, alpha_id));
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply,
                                        expm1_x_val_id, alpha_id,
                                        &alpha_times_expm1_x_val_id, 0));

  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_max,
                                        alpha_times_expm1_x_val_id, input_id,
                                        &output_id, 0));
  return ynn_status_success;
}

ynn_status define_leaky_relu(ynn_subgraph_t subgraph, uint32_t input_id,
                             float alpha, uint32_t& output_id) {
  uint32_t alpha_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(define_constant(subgraph, alpha, alpha_id));
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_leaky_relu,
                                        input_id, alpha_id, &output_id, 0));
  return ynn_status_success;
}

ynn_status define_hardswish(ynn_subgraph_t subgraph, uint32_t input_id,
                            uint32_t& output_id) {
  uint32_t x_div_6_id = YNN_INVALID_VALUE_ID;
  uint32_t one_sixth_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(define_constant(subgraph, 1.0f / 6.0f, one_sixth_id));
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply, input_id,
                                        one_sixth_id, &x_div_6_id, 0));

  uint32_t x_div_6_plus_0_5_id = YNN_INVALID_VALUE_ID;
  uint32_t half_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(define_constant(subgraph, 0.5f, half_id));
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_add, x_div_6_id,
                                        half_id, &x_div_6_plus_0_5_id, 0));

  uint32_t max_id = YNN_INVALID_VALUE_ID;
  uint32_t zero_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(define_constant(subgraph, 0.0f, zero_id));
  YNN_RETURN_IF_ERROR(ynn_define_binary(
      subgraph, ynn_binary_max, x_div_6_plus_0_5_id, zero_id, &max_id, 0));

  uint32_t relu6_id = YNN_INVALID_VALUE_ID;
  uint32_t one_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(define_constant(subgraph, 1.0f, one_id));
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_min, max_id,
                                        one_id, &relu6_id, 0));

  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply, input_id,
                                        relu6_id, &output_id, 0));
  return ynn_status_success;
}

}  // namespace ynn
