// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_DOT_H_
#define XNNPACK_YNNPACK_SUBGRAPH_DOT_H_

#include <cstdint>
#include <cstddef>

#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/dot/dot.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/runtime/buffer.h"

namespace ynn {

void define_transpose_a(ynn_subgraph& subgraph, ynn_node& node,
                        slinky::index_t tile_m, slinky::index_t tile_k,
                        int m_dim, uint32_t input_a_id, uint32_t output_id);

inline void define_transpose_a(ynn_subgraph& subgraph, ynn_node& node,
                               slinky::index_t tile_k, int m_dim,
                               uint32_t input_a_id, uint32_t output_id) {
  define_transpose_a(subgraph, node, /*tile_m=*/1, tile_k, m_dim, input_a_id,
                     output_id);
}

// Returns true if dots of type uint8 x `b_type` are faster than dots of type
// int8 x `b_type`.
bool prefer_uint8_dot(ynn_type b_type);

uint32_t define_pack_b(ynn_subgraph& subgraph, const dot_type& type,
                       const dot_kernel& kernel, size_t num_k_dims,
                       bool consistent_arithmetic, uint32_t input_b_id);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_DOT_H_
