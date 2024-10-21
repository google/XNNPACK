#include <stddef.h>
#include <stdint.h>

#include "xnnpack.h"

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

enum xnn_status xnn_define_static_mean(xnn_subgraph_t subgraph,
                                       size_t num_reduction_axes,
                                       const size_t* reduction_axes,
                                       uint32_t input_id, uint32_t output_id,
                                       uint32_t flags) {
  return xnn_define_static_reduce(subgraph, xnn_reduce_mean, num_reduction_axes,
                                  reduction_axes, input_id, output_id, flags);
}
