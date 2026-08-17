// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_COMPOSITES_COMPOSITES_H_
#define XNNPACK_YNNPACK_COMPOSITES_COMPOSITES_H_

#include <cstddef>
#include <cstdint>

#include "ynnpack/include/ynnpack.h"

namespace ynn {

// This header defines helpers for implementing common higher level operations
// using YNNPACK's lower level public API.

// gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
ynn_status define_gelu(ynn_subgraph_t subgraph, uint32_t input_id,
                       uint32_t& output_id);

// approx_gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
ynn_status define_approx_gelu(ynn_subgraph_t subgraph, uint32_t input_id,
                              uint32_t& output_id);

// elu(x) = x >= 0 ? x : alpha * (exp(x) - 1)
ynn_status define_elu(ynn_subgraph_t subgraph, uint32_t input_id, float alpha,
                      uint32_t& output_id);

// leaky_relu(x) = x >= 0 ? x : alpha * x
ynn_status define_leaky_relu(ynn_subgraph_t subgraph, uint32_t input_id,
                             float alpha, uint32_t& output_id);

// hardswish(x) = x * clamp(x / 6 + 0.5, 0, 1)
ynn_status define_hardswish(ynn_subgraph_t subgraph, uint32_t input_id,
                            uint32_t& output_id);

// softmax(x)_i = exp(beta * x_i) / sum_j(exp(beta * x_j))
ynn_status define_softmax(ynn_subgraph_t subgraph, uint32_t input_id,
                          float beta, uint32_t& output_id);

// log_softmax(x)_i = x_i - log(sum_j(exp(x_j)))
ynn_status define_log_softmax(ynn_subgraph_t subgraph, uint32_t input_id,
                              uint32_t& output_id);

// Computes average pooling of a 2D buffer. The `input_id` and `output_id`
// values must refer to rank 4 tensors. When averaging, the number of samples is
// the number of samples that are not padding.
ynn_status define_average_pool_2d(ynn_subgraph_t subgraph, uint32_t input_id,
                                  ynn_type type, bool padding_same,
                                  size_t filter_height, size_t filter_width,
                                  size_t stride_height, size_t stride_width,
                                  uint32_t& output_id);

// Computes a sum reduction, optionally dividing by the number of elements in
// the reduction if `mean` is true. The quantization parameters may be
// `YNN_INVALID_VALUE_ID`, indicating an identity value of 0 (for zero point) or
// 1 (for scale).
ynn_status define_reduce_sum(ynn_subgraph_t subgraph, size_t num_axes,
                             const int32_t* axes, uint32_t input_id,
                             uint32_t input_zero_point_id,
                             uint32_t input_scale_id, bool keep_dims, bool mean,
                             bool squared, ynn_type output_type,
                             uint32_t output_zero_point_id,
                             uint32_t output_scale_id, uint32_t& output_id);

// This function computes the quantization parameters of the result of a
// quantized dot operation. It computes the `zero_point` and `scale` values of
// the following equivalence:
//
//   (a.b - zero_point)*scale =
//       ((a - a_zero_point)*a_scale).(b - b_zerp_point)*b_scale
//
// It supports dynamic and static quantization parameters.
ynn_status define_dot_quantization(ynn_subgraph_t subgraph, size_t num_k_dims,
                                   uint32_t a_id, uint32_t a_zero_point_id,
                                   uint32_t a_scale_id, uint32_t b_id,
                                   uint32_t b_zero_point_id,
                                   uint32_t b_scale_id, uint32_t& zero_point_id,
                                   uint32_t& scale_id);

// Splits an f32 value into multiple bf16 values such that
// `sum(f32(output_ids[i])) ~= input_id`. The output values are ordered by
// descending significance.
ynn_status define_convert_f32_to_bf16_sum(ynn_subgraph_t subgraph,
                                          uint32_t input_id, size_t num_values,
                                          uint32_t* output_ids, uint32_t flags);

// Compute the dot of values that have been split into descending significance
// components. This computes the sum of the dot of the values in the cartesian
// product of a_ids x b_ids, where the sum of the index of the two values is
// not greater than `max_sum_index`.
ynn_status define_dot_sum(ynn_subgraph_t subgraph, size_t num_k_dims,
                          size_t num_a_values, const uint32_t* a_ids,
                          size_t num_b_values, const uint32_t* b_ids,
                          uint32_t input_c_id, uint32_t& output_id,
                          uint32_t flags, int max_sum_index = -1);

// Computes a blockwise quantized dot product of `a_id` and `b_id`.
// `a_id` has shape [..., M, K] (or [K] if rank 1).
// `b_id` has shape [..., K, N].
// `b_scale_id` is the scale tensor for `b_id` (either [..., N, num_blocks] or
// [..., num_blocks, 1, N]). `b_zero_point_id` is the zero point tensor for
// `b_id` (or YNN_INVALID_VALUE_ID). `a_zero_point_id` and `a_scale_id` are
// quantization parameters for `a_id` (or YNN_INVALID_VALUE_ID). If `a_id` is
// floating point and `a_scale_id` is YNN_INVALID_VALUE_ID, `a_id` is
// dynamically quantized to int8. `bias_id` is an optional bias tensor of shape
// [N] (or YNN_INVALID_VALUE_ID). `output_id` receives the result tensor.
ynn_status define_blockwise_dot(ynn_subgraph_t subgraph, uint32_t a_id,
                                uint32_t a_zero_point_id, uint32_t a_scale_id,
                                uint32_t b_id, uint32_t b_zero_point_id,
                                uint32_t b_scale_id, size_t block_size,
                                uint32_t bias_id, ynn_type output_type,
                                uint32_t& output_id, uint32_t flags = 0);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_COMPOSITES_COMPOSITES_H_
