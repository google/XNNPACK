// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_ELEMENTWISE_H_
#define XNNPACK_YNNPACK_SUBGRAPH_ELEMENTWISE_H_

#include <cstdint>

#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/binary/binary.h"
#include "ynnpack/kernels/ternary/ternary.h"
#include "ynnpack/subgraph/subgraph.h"

namespace ynn {

void define_binary(ynn_subgraph& subgraph, ynn_node& node, uint32_t input_a_id,
                   uint32_t input_b_id, uint32_t output_id,
                   ynn_binary_operator op, binary_kernel_fn kernel,
                   init_binary_params_fn init_params = nullptr);
void define_ternary(ynn_subgraph& subgraph, ynn_node& node, uint32_t input_a_id,
                    uint32_t input_b_id, uint32_t input_c_id,
                    uint32_t output_id, ternary_op op,
                    ternary_kernel_fn kernel);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_ELEMENTWISE_H_
