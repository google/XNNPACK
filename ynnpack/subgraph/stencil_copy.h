// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_STENCIL_COPY_H_
#define XNNPACK_YNNPACK_SUBGRAPH_STENCIL_COPY_H_

#include <cstddef>
#include <cstdint>

#include "ynnpack/subgraph/subgraph.h"

namespace ynn {

void define_stencil_copy(ynn_subgraph& subgraph, ynn_node& node,
                         ynn_node::stencil_copy op_data, uint32_t input_id,
                         uint32_t padding_id, uint32_t* output_id,
                         uint32_t flags);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_STENCIL_COPY_H_
