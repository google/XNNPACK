// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_STATIC_SLICE_H_
#define XNNPACK_YNNPACK_SUBGRAPH_STATIC_SLICE_H_

#include <cstdint>
#include <vector>

#include "ynnpack/subgraph/subgraph.h"

namespace ynn {

// `slices` should be sorted in ascending order by axis.
void define_static_slice(ynn_subgraph& subgraph, ynn_node& node,
                         uint32_t input_id, uint32_t output_id,
                         std::vector<ynn_node::static_slice::slice> slices,
                         bool slice_dims);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_STATIC_SLICE_H_
