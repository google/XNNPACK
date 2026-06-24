// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_COMPOSITES_COMPOSITES_H_
#define XNNPACK_YNNPACK_COMPOSITES_COMPOSITES_H_

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

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_COMPOSITES_COMPOSITES_H_
