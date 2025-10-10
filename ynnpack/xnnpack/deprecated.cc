// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "include/xnnpack.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/xnnpack/utils.h"
#include "ynnpack/xnnpack/xnnpack.h"

extern "C" {

xnn_status xnn_define_add2(xnn_subgraph_t subgraph, float output_min,
                           float output_max, uint32_t input1_id,
                           uint32_t input2_id, uint32_t output_id,
                           uint32_t flags) {
  xnn_binary_params params;
  params.output_min = output_min;
  params.output_max = output_max;
  return xnn_define_binary(subgraph, xnn_binary_add, &params, input1_id,
                           input2_id, output_id, flags);
}

xnn_status xnn_define_subtract(xnn_subgraph_t subgraph, float output_min,
                               float output_max, uint32_t input1_id,
                               uint32_t input2_id, uint32_t output_id,
                               uint32_t flags) {
  xnn_binary_params params;
  params.output_min = output_min;
  params.output_max = output_max;
  return xnn_define_binary(subgraph, xnn_binary_subtract, &params, input1_id,
                           input2_id, output_id, flags);
}

xnn_status xnn_define_multiply2(xnn_subgraph_t subgraph, float output_min,
                                float output_max, uint32_t input1_id,
                                uint32_t input2_id, uint32_t output_id,
                                uint32_t flags) {
  xnn_binary_params params;
  params.output_min = output_min;
  params.output_max = output_max;
  return xnn_define_binary(subgraph, xnn_binary_multiply, &params, input1_id,
                           input2_id, output_id, flags);
}

xnn_status xnn_define_divide(xnn_subgraph_t subgraph, float output_min,
                             float output_max, uint32_t input1_id,
                             uint32_t input2_id, uint32_t output_id,
                             uint32_t flags) {
  xnn_binary_params params;
  params.output_min = output_min;
  params.output_max = output_max;
  return xnn_define_binary(subgraph, xnn_binary_divide, &params, input1_id,
                           input2_id, output_id, flags);
}

xnn_status xnn_define_maximum2(xnn_subgraph_t subgraph, uint32_t input1_id,
                               uint32_t input2_id, uint32_t output_id,
                               uint32_t flags) {
  return xnn_define_binary(subgraph, xnn_binary_maximum, nullptr, input1_id,
                           input2_id, output_id, flags);
}

xnn_status xnn_define_minimum2(xnn_subgraph_t subgraph, uint32_t input1_id,
                               uint32_t input2_id, uint32_t output_id,
                               uint32_t flags) {
  return xnn_define_binary(subgraph, xnn_binary_minimum, nullptr, input1_id,
                           input2_id, output_id, flags);
}

xnn_status xnn_define_squared_difference(xnn_subgraph_t subgraph,
                                         uint32_t input1_id, uint32_t input2_id,
                                         uint32_t output_id, uint32_t flags) {
  return xnn_define_binary(subgraph, xnn_binary_squared_difference, nullptr,
                           input1_id, input2_id, output_id, flags);
}

xnn_status xnn_define_copysign(xnn_subgraph_t subgraph, uint32_t input1_id,
                               uint32_t input2_id, uint32_t output_id,
                               uint32_t flags) {
  return xnn_define_binary(subgraph, xnn_binary_copysign, nullptr, input1_id,
                           input2_id, output_id, flags);
}

xnn_status xnn_define_prelu(xnn_subgraph_t subgraph, uint32_t input1_id,
                            uint32_t input2_id, uint32_t output_id,
                            uint32_t flags) {
  return xnn_define_binary(subgraph, xnn_binary_prelu, nullptr, input1_id,
                           input2_id, output_id, flags);
}

xnn_status xnn_define_static_mean(xnn_subgraph_t subgraph,
                                  size_t num_reduction_axes,
                                  const size_t* reduction_axes,
                                  uint32_t input_id, uint32_t output_id,
                                  uint32_t flags) {
  return xnn_define_static_reduce(subgraph, xnn_reduce_mean, num_reduction_axes,
                                  reduction_axes, input_id, output_id, flags);
}

xnn_status xnn_define_global_average_pooling_1d(
    xnn_subgraph_t subgraph, float output_min, float output_max,
    uint32_t input_id, uint32_t output_id, uint32_t flags) {
  ynn_subgraph_t ynn_subgraph = subgraph->ynn;

  size_t reduction_axes[XNN_MAX_TENSOR_DIMS];

  reduction_axes[0] = ynn::rank_of_value(ynn_subgraph, input_id) - 2;

  xnn_status status =
      (xnn_define_static_reduce(subgraph, xnn_reduce_mean, 1, reduction_axes,
                                input_id, output_id, flags));

  if (status != xnn_status_success) {
    return status;
  }

  if (output_min != -INFINITY || output_max != INFINITY) {
    return ynn::xnn_status_from_ynn(
        ynn::implement_clamp(ynn_subgraph, output_min, output_max, output_id));
  }

  return xnn_status_success;
}

xnn_status xnn_define_global_average_pooling_2d(
    xnn_subgraph_t subgraph, float output_min, float output_max,
    uint32_t input_id, uint32_t output_id, uint32_t flags) {
  ynn_subgraph_t ynn_subgraph = subgraph->ynn;

  size_t reduction_axes[XNN_MAX_TENSOR_DIMS];

  reduction_axes[0] = ynn::rank_of_value(ynn_subgraph, input_id) - 3;
  reduction_axes[1] = ynn::rank_of_value(ynn_subgraph, input_id) - 2;

  xnn_status status = xnn_define_static_reduce(
      subgraph, xnn_reduce_mean, 2, reduction_axes, input_id, output_id, flags);

  if (status != xnn_status_success) {
    return status;
  }

  if (output_min != -INFINITY || output_max != INFINITY) {
    return ynn::xnn_status_from_ynn(
        ynn::implement_clamp(ynn_subgraph, output_min, output_max, output_id));
  }

  return xnn_status_success;
}

xnn_status xnn_define_global_sum_pooling_1d(xnn_subgraph_t subgraph,
                                            float output_min, float output_max,
                                            uint32_t input_id,
                                            uint32_t output_id,
                                            uint32_t flags) {
  ynn_subgraph_t ynn_subgraph = subgraph->ynn;

  size_t reduction_axes[XNN_MAX_TENSOR_DIMS];
  reduction_axes[0] = ynn::rank_of_value(ynn_subgraph, input_id) - 2;

  xnn_status status = xnn_define_static_reduce(
      subgraph, xnn_reduce_sum, 1, reduction_axes, input_id, output_id, flags);

  if (status != xnn_status_success) {
    return status;
  }

  if (output_min != -INFINITY || output_max != INFINITY) {
    return ynn::xnn_status_from_ynn(
        ynn::implement_clamp(ynn_subgraph, output_min, output_max, output_id));
  }

  return xnn_status_success;
}

xnn_status xnn_define_global_sum_pooling_2d(xnn_subgraph_t subgraph,
                                            float output_min, float output_max,
                                            uint32_t input_id,
                                            uint32_t output_id,
                                            uint32_t flags) {
  ynn_subgraph_t ynn_subgraph = subgraph->ynn;

  size_t reduction_axes[XNN_MAX_TENSOR_DIMS];
  reduction_axes[0] = ynn::rank_of_value(ynn_subgraph, input_id) - 3;
  reduction_axes[1] = ynn::rank_of_value(ynn_subgraph, input_id) - 2;

  xnn_status status = xnn_define_static_reduce(
      subgraph, xnn_reduce_sum, 2, reduction_axes, input_id, output_id, flags);

  if (status != xnn_status_success) {
    return status;
  }

  if (output_min != -INFINITY || output_max != INFINITY) {
    return ynn::xnn_status_from_ynn(
        ynn::implement_clamp(ynn_subgraph, output_min, output_max, output_id));
  }

  return xnn_status_success;
}

xnn_status xnn_define_convert(xnn_subgraph_t subgraph, uint32_t input_id,
                              uint32_t output_id, uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_convert, nullptr, input_id,
                          output_id, flags);
}

xnn_status xnn_define_abs(xnn_subgraph_t subgraph, uint32_t input_id,
                          uint32_t output_id, uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_abs, nullptr, input_id, output_id,
                          flags);
}

xnn_status xnn_define_bankers_rounding(xnn_subgraph_t subgraph,
                                       uint32_t input_id, uint32_t output_id,
                                       uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_bankers_rounding, nullptr,
                          input_id, output_id, flags);
}

xnn_status xnn_define_ceiling(xnn_subgraph_t subgraph, uint32_t input_id,
                              uint32_t output_id, uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_ceiling, nullptr, input_id,
                          output_id, flags);
}

xnn_status xnn_define_clamp(xnn_subgraph_t subgraph, float output_min,
                            float output_max, uint32_t input_id,
                            uint32_t output_id, uint32_t flags) {
  union xnn_unary_params params;
  params.clamp.min = output_min;
  params.clamp.max = output_max;
  return xnn_define_unary(subgraph, xnn_unary_clamp, &params, input_id,
                          output_id, flags);
}

xnn_status xnn_define_elu(xnn_subgraph_t subgraph, float alpha,
                          uint32_t input_id, uint32_t output_id,
                          uint32_t flags) {
  union xnn_unary_params params;
  params.elu.alpha = alpha;
  return xnn_define_unary(subgraph, xnn_unary_elu, &params, input_id, output_id,
                          flags);
}

xnn_status xnn_define_exp(xnn_subgraph_t subgraph, uint32_t input_id,
                          uint32_t output_id, uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_exp, nullptr, input_id, output_id,
                          flags);
}

xnn_status xnn_define_floor(xnn_subgraph_t subgraph, uint32_t input_id,
                            uint32_t output_id, uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_floor, nullptr, input_id,
                          output_id, flags);
}

xnn_status xnn_define_gelu(xnn_subgraph_t subgraph, uint32_t input_id,
                           uint32_t output_id, uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_gelu, nullptr, input_id,
                          output_id, flags);
}

xnn_status xnn_define_hardswish(xnn_subgraph_t subgraph, uint32_t input_id,
                                uint32_t output_id, uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_hardswish, nullptr, input_id,
                          output_id, flags);
}

xnn_status xnn_define_leaky_relu(xnn_subgraph_t subgraph, float negative_slope,
                                 uint32_t input_id, uint32_t output_id,
                                 uint32_t flags) {
  union xnn_unary_params params;
  params.leaky_relu.negative_slope = negative_slope;
  return xnn_define_unary(subgraph, xnn_unary_leaky_relu, &params, input_id,
                          output_id, flags);
}

xnn_status xnn_define_log(xnn_subgraph_t subgraph, uint32_t input_id,
                          uint32_t output_id, uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_log, nullptr, input_id, output_id,
                          flags);
}

xnn_status xnn_define_negate(xnn_subgraph_t subgraph, uint32_t input_id,
                             uint32_t output_id, uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_negate, nullptr, input_id,
                          output_id, flags);
}

xnn_status xnn_define_sigmoid(xnn_subgraph_t subgraph, uint32_t input_id,
                              uint32_t output_id, uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_sigmoid, nullptr, input_id,
                          output_id, flags);
}

xnn_status xnn_define_square(xnn_subgraph_t subgraph, uint32_t input_id,
                             uint32_t output_id, uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_square, nullptr, input_id,
                          output_id, flags);
}

xnn_status xnn_define_square_root(xnn_subgraph_t subgraph, uint32_t input_id,
                                  uint32_t output_id, uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_square_root, nullptr, input_id,
                          output_id, flags);
}

xnn_status xnn_define_reciprocal_square_root(xnn_subgraph_t subgraph,
                                             uint32_t input_id,
                                             uint32_t output_id,
                                             uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_reciprocal_square_root, nullptr,
                          input_id, output_id, flags);
}

xnn_status xnn_define_tanh(xnn_subgraph_t subgraph, uint32_t input_id,
                           uint32_t output_id, uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_tanh, nullptr, input_id,
                          output_id, flags);
}

xnn_status xnn_define_concatenate2(xnn_subgraph_t subgraph, int32_t axis,
                                   uint32_t input1_id, uint32_t input2_id,
                                   uint32_t output_id, uint32_t flags) {
  const uint32_t inputs_id[2] = {input1_id, input2_id};
  return xnn_define_concatenate(subgraph, axis, /*num_inputs=*/2, inputs_id,
                                output_id, flags);
}

xnn_status xnn_define_concatenate3(xnn_subgraph_t subgraph, int32_t axis,
                                   uint32_t input1_id, uint32_t input2_id,
                                   uint32_t input3_id, uint32_t output_id,
                                   uint32_t flags) {
  const uint32_t inputs_id[3] = {input1_id, input2_id, input3_id};
  return xnn_define_concatenate(subgraph, axis, /*num_inputs=*/3, inputs_id,
                                output_id, flags);
}

xnn_status xnn_define_concatenate4(xnn_subgraph_t subgraph, int32_t axis,
                                   uint32_t input1_id, uint32_t input2_id,
                                   uint32_t input3_id, uint32_t input4_id,
                                   uint32_t output_id, uint32_t flags) {
  const uint32_t inputs_id[4] = {input1_id, input2_id, input3_id, input4_id};
  return xnn_define_concatenate(subgraph, axis, /*num_inputs=*/4, inputs_id,
                                output_id, flags);
}

xnn_status xnn_define_concatenate5(xnn_subgraph_t subgraph, int32_t axis,
                                   uint32_t input1_id, uint32_t input2_id,
                                   uint32_t input3_id, uint32_t input4_id,
                                   uint32_t input5_id, uint32_t output_id,
                                   uint32_t flags) {
  const uint32_t inputs_id[5] = {input1_id, input2_id, input3_id, input4_id,
                                 input5_id};
  return xnn_define_concatenate(subgraph, axis, /*num_inputs=*/5, inputs_id,
                                output_id, flags);
}

xnn_status xnn_define_even_split2(xnn_subgraph_t subgraph, int32_t split_dim,
                                  uint32_t input_id, uint32_t output1_id,
                                  uint32_t output2_id, uint32_t flags) {
  const uint32_t outputs_id[2] = {output1_id, output2_id};
  return xnn_define_even_split(subgraph, split_dim, input_id, /*num_outputs=*/2,
                               outputs_id, flags);
}

xnn_status xnn_define_even_split3(xnn_subgraph_t subgraph, int32_t split_dim,
                                  uint32_t input_id, uint32_t output1_id,
                                  uint32_t output2_id, uint32_t output3_id,
                                  uint32_t flags) {
  const uint32_t outputs_id[3] = {output1_id, output2_id, output3_id};
  return xnn_define_even_split(subgraph, split_dim, input_id, /*num_outputs=*/3,
                               outputs_id, flags);
}

xnn_status xnn_define_even_split4(xnn_subgraph_t subgraph, int32_t split_dim,
                                  uint32_t input_id, uint32_t output1_id,
                                  uint32_t output2_id, uint32_t output3_id,
                                  uint32_t output4_id, uint32_t flags) {
  const uint32_t outputs_id[4] = {output1_id, output2_id, output3_id,
                                  output4_id};
  return xnn_define_even_split(subgraph, split_dim, input_id, /*num_outputs=*/4,
                               outputs_id, flags);
}

}  // extern "C"
