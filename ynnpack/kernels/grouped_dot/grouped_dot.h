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
// - m_start: Starting token index for this slice.
// - m_count: Number of tokens for this slice.
// - n_start: Starting output channel index for this slice.
// - n_count: Number of output channels for this slice.
// - D_in: Input dimension.
// - D_out: Output dimension.
// - a_stride_m: Stride in bytes between rows (tokens) in input_a.
// - c_stride_m: Stride in bytes between rows (tokens) in output.
// - packed_b_stride: Stride in bytes of the tiles_k dimension in packed_b.
// - expert_packed_stride: Stride in bytes between experts in the packed_b
//   buffer.
// - kernel: The underlying dot kernel to use.
void grouped_dot(size_t E, const int32_t* expert_counts,
                 const int32_t* offsets, const void* input_a,
                 const void* input_b, void* output, size_t m_start,
                 size_t m_count, size_t n_start, size_t n_count,
                 size_t D_in, size_t D_out, size_t a_stride_m,
                 size_t c_stride_m, size_t packed_b_stride,
                 size_t expert_packed_stride, const dot_kernel& kernel,
                 size_t block_n, size_t blocks_n_stride, size_t elem_size_a,
                 size_t elem_size_c);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_GROUPED_DOT_GROUPED_DOT_H_
