// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_UTILS_H_
#define XNNPACK_YNNPACK_SUBGRAPH_UTILS_H_

#include <cstdint>

#include "ynnpack/subgraph/subgraph.h"
#include "slinky/runtime/buffer.h"

namespace ynn {

// Determine if we can compute an input and an output value in place or not.
bool allow_in_place(uint32_t input_id, uint32_t output_id,
                    const ynn_subgraph& subgraph);

uint32_t define_transpose_a(ynn_subgraph& subgraph, slinky::index_t tile_k,
                            uint32_t input_a_id);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_UTILS_H_
