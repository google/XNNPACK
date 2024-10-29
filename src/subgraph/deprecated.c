#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack.h"
#include "xnnpack/subgraph.h"

enum xnn_status xnn_define_add2(xnn_subgraph_t subgraph, float output_min,
                                float output_max, uint32_t input1_id,
                                uint32_t input2_id, uint32_t output_id,
                                uint32_t flags) {
  struct xnn_binary_params params;
  params.output_min = output_min;
  params.output_max = output_max;
  return xnn_define_binary(subgraph, xnn_binary_add, &params, input1_id,
                           input2_id, output_id, flags);
}

enum xnn_status xnn_define_subtract(xnn_subgraph_t subgraph, float output_min,
                                    float output_max, uint32_t input1_id,
                                    uint32_t input2_id, uint32_t output_id,
                                    uint32_t flags) {
  struct xnn_binary_params params;
  params.output_min = output_min;
  params.output_max = output_max;
  return xnn_define_binary(subgraph, xnn_binary_subtract, &params, input1_id,
                           input2_id, output_id, flags);
}

enum xnn_status xnn_define_multiply2(xnn_subgraph_t subgraph, float output_min,
                                     float output_max, uint32_t input1_id,
                                     uint32_t input2_id, uint32_t output_id,
                                     uint32_t flags) {
  struct xnn_binary_params params;
  params.output_min = output_min;
  params.output_max = output_max;
  return xnn_define_binary(subgraph, xnn_binary_multiply, &params, input1_id,
                           input2_id, output_id, flags);
}

enum xnn_status xnn_define_divide(xnn_subgraph_t subgraph, float output_min,
                                  float output_max, uint32_t input1_id,
                                  uint32_t input2_id, uint32_t output_id,
                                  uint32_t flags) {
  struct xnn_binary_params params;
  params.output_min = output_min;
  params.output_max = output_max;
  return xnn_define_binary(subgraph, xnn_binary_divide, &params, input1_id,
                           input2_id, output_id, flags);
}

enum xnn_status xnn_define_maximum2(xnn_subgraph_t subgraph, uint32_t input1_id,
                                    uint32_t input2_id, uint32_t output_id,
                                    uint32_t flags) {
  return xnn_define_binary(subgraph, xnn_binary_maximum, NULL, input1_id,
                           input2_id, output_id, flags);
}

enum xnn_status xnn_define_minimum2(xnn_subgraph_t subgraph, uint32_t input1_id,
                                    uint32_t input2_id, uint32_t output_id,
                                    uint32_t flags) {
  return xnn_define_binary(subgraph, xnn_binary_minimum, NULL, input1_id,
                           input2_id, output_id, flags);
}

enum xnn_status xnn_define_squared_difference(xnn_subgraph_t subgraph,
                                              uint32_t input1_id,
                                              uint32_t input2_id,
                                              uint32_t output_id,
                                              uint32_t flags) {
  return xnn_define_binary(subgraph, xnn_binary_squared_difference, NULL,
                           input1_id, input2_id, output_id, flags);
}

enum xnn_status xnn_define_copysign(xnn_subgraph_t subgraph, uint32_t input1_id,
                                    uint32_t input2_id, uint32_t output_id,
                                    uint32_t flags) {
  return xnn_define_binary(subgraph, xnn_binary_copysign, NULL, input1_id,
                           input2_id, output_id, flags);
}

enum xnn_status xnn_define_prelu(xnn_subgraph_t subgraph, uint32_t input1_id,
                                 uint32_t input2_id, uint32_t output_id,
                                 uint32_t flags) {
  return xnn_define_binary(subgraph, xnn_binary_prelu, NULL,
                           input1_id, input2_id, output_id, flags);
}

enum xnn_status xnn_define_static_mean(xnn_subgraph_t subgraph,
                                       size_t num_reduction_axes,
                                       const size_t* reduction_axes,
                                       uint32_t input_id, uint32_t output_id,
                                       uint32_t flags) {
  return xnn_define_static_reduce(subgraph, xnn_reduce_mean, num_reduction_axes,
                                  reduction_axes, input_id, output_id, flags);
}

enum xnn_status xnn_define_global_average_pooling_1d(
  xnn_subgraph_t subgraph,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  const struct xnn_value* input_value = &subgraph->values[input_id];

  size_t reduction_axes[XNN_MAX_TENSOR_DIMS];

  reduction_axes[0] = input_value->shape.num_dims - 2;

  enum xnn_status status = (xnn_define_static_reduce(
    subgraph, xnn_reduce_mean, 1, reduction_axes, input_id,
    output_id, flags));

  if (status != xnn_status_success) {
    return status;
  }

  if (output_min != -INFINITY || output_max != INFINITY) {
    return xnn_insert_clamp_node(subgraph, output_min, output_max,
                                 &subgraph->nodes[subgraph->num_nodes - 1]);
  }

  return xnn_status_success;
}

enum xnn_status xnn_define_global_average_pooling_2d(
  xnn_subgraph_t subgraph,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  const struct xnn_value* input_value = &subgraph->values[input_id];

  size_t reduction_axes[XNN_MAX_TENSOR_DIMS];

  reduction_axes[0] = input_value->shape.num_dims - 3;
  reduction_axes[1] = input_value->shape.num_dims - 2;

  enum xnn_status status = xnn_define_static_reduce(
    subgraph, xnn_reduce_mean, 2, reduction_axes, input_id,
    output_id, flags);

  if (status != xnn_status_success) {
    return status;
  }

  if (output_min != -INFINITY || output_max != INFINITY) {
    return xnn_insert_clamp_node(subgraph, output_min, output_max,
                                 &subgraph->nodes[subgraph->num_nodes - 1]);
  }

  return xnn_status_success;
}

enum xnn_status xnn_define_global_sum_pooling_1d(
  xnn_subgraph_t subgraph,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  const struct xnn_value* input_value = &subgraph->values[input_id];
  size_t reduction_axes[XNN_MAX_TENSOR_DIMS];
  reduction_axes[0] = input_value->shape.num_dims - 2;

  enum xnn_status status = xnn_define_static_reduce(
    subgraph, xnn_reduce_sum, 1, reduction_axes, input_id,
    output_id, flags);

  if (status != xnn_status_success) {
    return status;
  }

  if (output_min != -INFINITY || output_max != INFINITY) {
    return xnn_insert_clamp_node(subgraph, output_min, output_max,
                                 &subgraph->nodes[subgraph->num_nodes - 1]);
  }

  return xnn_status_success;
}

enum xnn_status xnn_define_global_sum_pooling_2d(
  xnn_subgraph_t subgraph,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  const struct xnn_value* input_value = &subgraph->values[input_id];
  size_t reduction_axes[XNN_MAX_TENSOR_DIMS];
  reduction_axes[0] = input_value->shape.num_dims - 3;
  reduction_axes[1] = input_value->shape.num_dims - 2;

  enum xnn_status status = xnn_define_static_reduce(
    subgraph, xnn_reduce_sum, 2, reduction_axes, input_id,
    output_id, flags);

  if (status != xnn_status_success) {
    return status;
  }

  if (output_min != -INFINITY || output_max != INFINITY) {
    return xnn_insert_clamp_node(subgraph, output_min, output_max,
                                 &subgraph->nodes[subgraph->num_nodes - 1]);
  }

  return xnn_status_success;
}

enum xnn_status xnn_define_convert(xnn_subgraph_t subgraph, uint32_t input_id,
                                   uint32_t output_id, uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_convert, NULL, input_id, output_id,
                          flags);
}

enum xnn_status xnn_define_abs(xnn_subgraph_t subgraph, uint32_t input_id,
                               uint32_t output_id, uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_abs, NULL, input_id, output_id, 
                          flags);
}

enum xnn_status xnn_define_bankers_rounding(xnn_subgraph_t subgraph,
                                            uint32_t input_id,
                                            uint32_t output_id,
                                            uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_bankers_rounding, NULL, input_id,
                          output_id, flags);
}

enum xnn_status xnn_define_ceiling(xnn_subgraph_t subgraph, uint32_t input_id,
                                   uint32_t output_id, uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_ceiling, NULL, input_id, output_id,
                          flags);
}

enum xnn_status xnn_define_clamp(xnn_subgraph_t subgraph, float output_min,
                                 float output_max, uint32_t input_id,
                                 uint32_t output_id, uint32_t flags) {
  union xnn_unary_params params;
  params.clamp.min = output_min;
  params.clamp.max = output_max;
  return xnn_define_unary(subgraph, xnn_unary_clamp, &params, input_id, output_id,
                          flags);
}

enum xnn_status xnn_define_elu(xnn_subgraph_t subgraph, float alpha,
                               uint32_t input_id, uint32_t output_id,
                               uint32_t flags) {
  union xnn_unary_params params;
  params.elu.alpha = alpha;
  return xnn_define_unary(subgraph, xnn_unary_elu, &params, input_id, output_id,
                          flags);
}

enum xnn_status xnn_define_exp(xnn_subgraph_t subgraph, uint32_t input_id,
                               uint32_t output_id, uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_exp, NULL, input_id, output_id,
                          flags);
}

enum xnn_status xnn_define_floor(xnn_subgraph_t subgraph, uint32_t input_id,
                                 uint32_t output_id, uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_floor, NULL, input_id, output_id,
                          flags);
}

enum xnn_status xnn_define_gelu(xnn_subgraph_t subgraph, uint32_t input_id,
                                uint32_t output_id, uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_gelu, NULL, input_id, output_id,
                          flags);
}

enum xnn_status xnn_define_hardswish(xnn_subgraph_t subgraph, uint32_t input_id,
                                     uint32_t output_id, uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_hardswish, NULL, input_id, output_id,
                          flags);
}

enum xnn_status xnn_define_leaky_relu(xnn_subgraph_t subgraph,
                                      float negative_slope, uint32_t input_id,
                                      uint32_t output_id, uint32_t flags) {
  union xnn_unary_params params;
  params.leaky_relu.negative_slope = negative_slope;
  return xnn_define_unary(subgraph, xnn_unary_leaky_relu, &params, input_id, output_id,
                          flags);
}

enum xnn_status xnn_define_log(xnn_subgraph_t subgraph, uint32_t input_id,
                               uint32_t output_id, uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_log, NULL, input_id, output_id,
                          flags);
}

enum xnn_status xnn_define_negate(xnn_subgraph_t subgraph, uint32_t input_id,
                                  uint32_t output_id, uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_negate, NULL, input_id, output_id,
                          flags);
}

enum xnn_status xnn_define_sigmoid(xnn_subgraph_t subgraph, uint32_t input_id,
                                   uint32_t output_id, uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_sigmoid, NULL, input_id, output_id,
                          flags);
}

enum xnn_status xnn_define_square(xnn_subgraph_t subgraph, uint32_t input_id,
                                  uint32_t output_id, uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_square, NULL, input_id, output_id,
                          flags);
}

enum xnn_status xnn_define_square_root(xnn_subgraph_t subgraph,
                                       uint32_t input_id, uint32_t output_id,
                                       uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_square_root, NULL, input_id, output_id,
                          flags);
}

enum xnn_status xnn_define_reciprocal_square_root(xnn_subgraph_t subgraph,
                                                  uint32_t input_id,
                                                  uint32_t output_id,
                                                  uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_reciprocal_square_root, NULL, input_id,
                          output_id, flags);
}

enum xnn_status xnn_define_tanh(xnn_subgraph_t subgraph, uint32_t input_id,
                                uint32_t output_id, uint32_t flags) {
  return xnn_define_unary(subgraph, xnn_unary_tanh, NULL, input_id, output_id,
                          flags);
}
