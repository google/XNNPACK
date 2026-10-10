// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_GROUPED_DOT_GROUPED_DOT_H_
#define XNNPACK_YNNPACK_KERNELS_GROUPED_DOT_GROUPED_DOT_H_

#include <cstddef>
#include <cstdint>

#include "ynnpack/kernels/dot/dot.h"

namespace ynn {

// Grouped dot kernel computes:
//   C_out_slice = A_slice * B_slice
// for each expert, where B_slice is the packed weights for that expert.
//
// Arguments:
// - E: Number of experts.
// - expert_counts: Array of size E with token counts per expert.
// - offsets: Array of size E+1 with token offsets in input_a/output.
// - input_a: Dispatched tokens of shape [NK, D_in].
// - input_b: Packed weights for all experts.
// - output: Output buffer of shape [NK, D_out].
// - D_in: Input dimension.
// - D_out: Output dimension.
// - packed_b_stride: Stride in bytes between experts in the packed_b buffer.
// - kernel: The underlying dot kernel to use.
void grouped_dot(size_t E, const int32_t* expert_counts, const int32_t* offsets,
                 const float* input_a, const void* input_b, float* output,
                 size_t D_in, size_t D_out, size_t packed_b_stride,
                 const dot_kernel& kernel);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_GROUPED_DOT_GROUPED_DOT_H_
