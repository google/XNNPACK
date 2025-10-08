// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_XNNPACK_UTILS_H_
#define XNNPACK_YNNPACK_XNNPACK_UTILS_H_

#include <stdint.h>

#include <cstddef>

#include "include/xnnpack.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

xnn_status xnn_status_from_ynn(ynn_status status);

uint32_t value_flags_from_xnn(uint32_t flags);

ynn_unary_operator unary_operator_from_xnn(xnn_unary_operator op);
ynn_binary_operator binary_operator_from_xnn(xnn_binary_operator op);
ynn_reduce_operator reduce_operator_from_xnn(xnn_reduce_operator op);
ynn_type type_from_xnn(xnn_datatype type);
xnn_datatype xnn_datatype_from_ynn(ynn_type type);

// Define a new tensor of rank `rank` that has a similar type as `type_id`.
ynn_status define_tensor_value_like(ynn_subgraph_t subgraph, uint32_t type_id,
                                    size_t rank, uint32_t* id_out);
// Define a new tensor that has a similar type and shape as `id`.
ynn_status define_tensor_value_like(ynn_subgraph_t subgraph, uint32_t id,
                                    uint32_t* id_out);
// Define a new scalar-valued tensor that has the same type as `id`.
ynn_status define_scalar_value_like(ynn_subgraph_t subgraph, uint32_t id,
                                    float value_fp32, uint32_t* id_out);

// Get the type that should be used to compute a sum reduction for `type.
ynn_type accumulator_for_type(ynn_type type);

// Define a dot operation that supports the various combinations of types and
// quantization that XNNPACK does.
ynn_status define_xnn_dot(ynn_subgraph_t subgraph, size_t num_k_dims,
                          uint32_t a_id, uint32_t b_id, uint32_t bias_id,
                          uint32_t output_id);

ynn_status define_binary_scalar_a(ynn_subgraph_t subgraph,
                                  ynn_binary_operator op, float scalar_a,
                                  uint32_t input_b_id, uint32_t* output_id);
ynn_status define_binary_scalar_b(ynn_subgraph_t subgraph,
                                  ynn_binary_operator op, uint32_t input_a_id,
                                  float scalar_b, uint32_t* output_id);

// Insert broadcast_like operations to emulate XNNPACK implicit broadcasting.
ynn_status implement_xnn_broadcasting(ynn_subgraph_t subgraph,
                                      uint32_t* input_a_id,
                                      uint32_t* input_b_id, uint32_t flags = 0,
                                      size_t exclude_a = 0,
                                      size_t exclude_b = 0);

ynn_status define_binary_with_broadcasting(
    ynn_subgraph_t subgraph, ynn_binary_operator op, uint32_t input_a_id,
    uint32_t input_b_id, uint32_t* output_id, uint32_t flags = 0);

// Implements (x / 2) * (1 + erf(x * sqrt(2) / 2))
ynn_status implement_gelu(ynn_subgraph_t subgraph, uint32_t input_id,
                          uint32_t output_id);

// Make a clamp operation.
ynn_status define_clamp(ynn_subgraph_t subgraph, float min, float max,
                        uint32_t input_id, uint32_t* output_id);

// Replace output_id with a clamped version of that tensor.
ynn_status implement_clamp(ynn_subgraph_t subgraph, float min, float max,
                           uint32_t output_id);

ynn_status define_xnn_stencil(
    ynn_subgraph_t subgraph, uint32_t input_padding_top,
    uint32_t input_padding_right, uint32_t input_padding_bottom,
    uint32_t input_padding_left, float padding_value, uint32_t pooling_height,
    uint32_t pooling_width, uint32_t stride_height, uint32_t stride_width,
    uint32_t dilation_height, uint32_t dilation_width, uint32_t input_id,
    uint32_t* stencil_id, uint32_t flags);

ynn_type type_of_value(ynn_subgraph_t subgraph, uint32_t id);
size_t rank_of_value(ynn_subgraph_t subgraph, uint32_t id);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_XNNPACK_UTILS_H_
