// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_DOT_H_
#define XNNPACK_YNNPACK_SUBGRAPH_DOT_H_

#include <cstdint>
#include <functional>
#include <tuple>

#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"

struct ynn_runtime;

namespace ynn {

void define_transpose_a(ynn_subgraph& subgraph, ynn_node& node,
                        slinky::index_t tile_k, int m_dim, uint32_t input_a_id,
                        uint32_t output_id);

// Returns true if dots of type uint8 x `b_type` are faster than dots of type
// int8 x `b_type`.
bool prefer_uint8_dot(ynn_type b_type);

using choose_split_factors_fn =
    std::function<std::tuple<slinky::expr, slinky::expr, slinky::expr>(
        ::ynn_runtime& runtime, slinky::expr m, slinky::expr n, slinky::expr k,
        slinky::expr block_n)>;

extern choose_split_factors_fn choose_split_factors_hook;

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_DOT_H_
