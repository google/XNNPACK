// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_ELEMENTWISE_H_
#define XNNPACK_YNNPACK_SUBGRAPH_ELEMENTWISE_H_

#include <cstdint>

#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/binary/binary.h"
#include "ynnpack/kernels/dequantize_dot/dequantize_dot.h"
#include "ynnpack/kernels/ternary/ternary.h"
#include "ynnpack/kernels/unary/unary.h"
#include "ynnpack/subgraph/subgraph.h"

namespace ynn {

void define_unary(ynn_subgraph& subgraph, ynn_node& node, uint32_t input_a_id,
                  uint32_t output_id, ynn_unary_operator op,
                  unary_kernel_fn kernel);
void define_binary(ynn_subgraph& subgraph, ynn_node& node, uint32_t input_a_id,
                   uint32_t input_b_id, uint32_t output_id,
                   ynn_binary_operator op, binary_kernel_fn kernel);
void define_ternary(ynn_subgraph& subgraph, ynn_node& node, uint32_t input_a_id,
                    uint32_t input_b_id, uint32_t input_c_id,
                    uint32_t output_id, ternary_op op,
                    ternary_kernel_fn kernel);
void define_lut(ynn_subgraph& subgraph, ynn_node& node, uint32_t input_id,
                uint32_t lut_id, uint32_t& output_id);

void define_dequantize_dot(ynn_subgraph& subgraph, ynn_node& node,
                           ynn_type output_type, uint32_t dot_id,
                           uint32_t a_offset_id, uint32_t b_offset_id,
                           uint32_t a_scale_id, uint32_t b_scale_id,
                           uint32_t offset_id, uint32_t& output_id,
                           const dequantize_dot_params& params);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_ELEMENTWISE_H_
