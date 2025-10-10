// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_XNNPACK_DYNAMIC_QUANTIZATION_H_
#define XNNPACK_YNNPACK_XNNPACK_DYNAMIC_QUANTIZATION_H_

#include <cstddef>
#include <cstdint>

#include "ynnpack/include/ynnpack.h"

namespace ynn {

// Defines the subgraph to produce the scale and zero point tensors of the
// output.
ynn_status compute_qd8_params(ynn_subgraph_t subgraph, size_t num_nonbatch_axes,
                              uint32_t input_id, uint32_t output_id);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_XNNPACK_DYNAMIC_QUANTIZATION_H_
