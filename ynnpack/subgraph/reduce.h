// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_REDUCE_H_
#define XNNPACK_YNNPACK_SUBGRAPH_REDUCE_H_

#include <cstdint>

#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"

namespace ynn {

void define_reduce(ynn_subgraph& subgraph, ynn_node& node,
                   ynn_reduce_operator op, const ynn::axes_set& k_dims,
                   uint32_t input_a_id, uint32_t input_b_id, uint32_t output_id,
                   bool keep_dims);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_REDUCE_H_
