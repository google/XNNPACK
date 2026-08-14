// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/grouped_dot/grouped_dot.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/kernels/dot/dot.h"

namespace ynn {

void grouped_dot(size_t E, const int32_t* expert_counts,
                 const int32_t* offsets, const void* input_a,
                 const void* input_b, void* output, size_t m_start,
                 size_t m_count, size_t n_start, size_t n_count, size_t D_in,
                 size_t D_out, size_t a_stride_m, size_t c_stride_m,
                 size_t packed_b_stride, size_t expert_packed_stride,
                 const dot_kernel& kernel, size_t block_n,
                 size_t blocks_n_stride, size_t elem_size_a,
                 size_t elem_size_c) {
  assert(kernel.kernel != nullptr);
  if (m_count == 0 || n_count == 0) return;

  size_t tile_k = kernel.tile_k;
  size_t block_m = kernel.block_m;

  // Stride of B in bytes to move by tile_k elements in reduction dimension.
  // packed_b_stride is input_b.dim(2).stride() in Slinky, which is the stride
  // of tiles_k. The kernel expects stride in bytes divided by tile_k.
  size_t b_stride_k1 = packed_b_stride / tile_k;

  // D_in must be a multiple of tile_k for the optimized kernel.
  assert(D_in % tile_k == 0);

  size_t m_end = m_start + m_count;
  size_t n_end = n_start + n_count;

  // Find first expert overlapping [m_start, m_end)
  auto it =
      std::upper_bound(offsets, offsets + E, static_cast<int32_t>(m_start));
  size_t e_start = (it == offsets) ? 0 : std::distance(offsets, it) - 1;

  for (size_t e = e_start; e < E; ++e) {
    int32_t exp_offset = offsets[e];
    if (static_cast<size_t>(exp_offset) >= m_end) {
      break;
    }
    int32_t exp_count = expert_counts[e];
    if (exp_count == 0) continue;

    size_t exp_start = static_cast<size_t>(exp_offset);
    size_t exp_end = exp_start + static_cast<size_t>(exp_count);

    size_t cur_start = std::max(m_start, exp_start);
    size_t cur_end = std::min(m_end, exp_end);
    if (cur_start >= cur_end) continue;

    size_t cur_m_total = cur_end - cur_start;
    const void* B_e = offset_bytes(input_b, e * expert_packed_stride);

    // Loop over M in steps of block_m.
    for (size_t m_offset = 0; m_offset < cur_m_total; m_offset += block_m) {
      size_t current_m = std::min(cur_m_total - m_offset, block_m);
      size_t token_idx = cur_start + m_offset;
      size_t m_rel = token_idx - m_start;

      const void* A_step = offset_bytes(input_a, m_rel * a_stride_m);
      void* C_step = offset_bytes(output, m_rel * c_stride_m);

      // Loop over N in steps of block_n (packing block_n)
      size_t n_start_aligned = (n_start / block_n) * block_n;
      const void* B_step =
          offset_bytes(B_e, (n_start_aligned / block_n) * blocks_n_stride);
      for (size_t n_offset = n_start_aligned; n_offset < n_end;
           n_offset += block_n) {
        size_t current_n = std::min(n_end - n_offset, block_n);
        ptrdiff_t n_rel = static_cast<ptrdiff_t>(n_offset) -
                          static_cast<ptrdiff_t>(n_start);
        void* C_n_step = offset_bytes(C_step, n_rel * elem_size_c);
        kernel.kernel(current_m, current_n, 1, 1, D_in, a_stride_m, 0, 0,
                      A_step, 0, 0, b_stride_k1, B_step, 0, nullptr,
                      c_stride_m, C_n_step);
        B_step = static_cast<const char*>(B_step) + blocks_n_stride;
      }
    }
  }
}

}  // namespace ynn
