// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_UTILS_H_
#define XNNPACK_YNNPACK_SUBGRAPH_UTILS_H_

#include <cstdint>
#include <memory>

#include "ynnpack/subgraph/subgraph.h"

namespace ynn {

// Determine if we can compute an input and an output value in place or not.
bool allow_in_place(uint32_t input_id, uint32_t output_id,
                    const ynn_subgraph& subgraph);

// Clone a subset of the subgraph that is required to compute `output_id` from
// `input_id`. `input_id` in the new subgraph will be an external input.
// All other inputs required to compute `output_id` (and not derived from
// `input_id`) will also become external inputs in the new subgraph.
// Ids of values in the new subgraph will be different from the original
// subgraph.
// Returns nullptr if the subgraph does not have a path from `input_id` to
// `output_id`.
std::unique_ptr<ynn_subgraph> clone_subgraph_subset(
    const ynn_subgraph& subgraph, uint32_t input_id, uint32_t output_id,
    uint32_t& cloned_input_id, uint32_t& cloned_output_id);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_UTILS_H_
