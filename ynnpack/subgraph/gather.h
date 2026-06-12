// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_GATHER_H_
#define XNNPACK_YNNPACK_SUBGRAPH_GATHER_H_

#include <cstdint>

#include "ynnpack/subgraph/subgraph.h"

namespace ynn {

void define_gather(ynn_subgraph& subgraph, ynn_node& node, int32_t axis,
                   uint32_t input_id, uint32_t index_id, uint32_t& output_id);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_GATHER_H_
