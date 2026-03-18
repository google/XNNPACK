// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_DOT_H_
#define XNNPACK_YNNPACK_SUBGRAPH_DOT_H_

#include <cstdint>

#include "ynnpack/subgraph/subgraph.h"
#include "slinky/runtime/buffer.h"

namespace ynn {

void define_transpose_a(ynn_subgraph& subgraph, ynn_node& node,
                        slinky::index_t tile_k, int32_t m_dim,
                        uint32_t input_a_id, uint32_t output_id);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_DOT_H_
