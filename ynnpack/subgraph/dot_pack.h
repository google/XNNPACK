// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_DOT_PACK_H_
#define XNNPACK_YNNPACK_SUBGRAPH_DOT_PACK_H_

#include <cstddef>
#include <cstdint>

#include "ynnpack/kernels/dot/dot.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/runtime/buffer.h"

namespace ynn {

uint32_t define_pack_b(ynn_subgraph& subgraph, const dot_type& type,
                       const dot_kernel& kernel, size_t num_k_dims,
                       uint32_t input_b_id, bool transpose = false);

void define_transpose_a(ynn_subgraph& subgraph, ynn_node& node,
                        slinky::index_t tile_m, slinky::index_t tile_k,
                        int m_dim, uint32_t input_a_id, uint32_t output_id);

uint32_t define_transpose_a(ynn_subgraph& subgraph, slinky::index_t tile_m,
                            slinky::index_t tile_k, int32_t m_dim,
                            uint32_t input_a_id);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_DOT_PACK_H_
