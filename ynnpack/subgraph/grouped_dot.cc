// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/grouped_dot.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "ynnpack/kernels/dot/dot.h"

namespace ynn {

namespace {

// Helper to offset pointers in bytes.
template <typename T>
T* offset_bytes(T* ptr, ptrdiff_t offset) {
  return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr) + offset);
}

template <typename T>
const T* offset_bytes(const T* ptr, ptrdiff_t offset) {
  return reinterpret_cast<const T*>(reinterpret_cast<const char*>(ptr) +
                                    offset);
}

}  // namespace

void grouped_dot(size_t E, const int32_t* expert_counts, const int32_t* offsets,
                 const float* input_a, const void* input_b, float* output,
                 size_t D_in, size_t D_out, size_t packed_b_stride,
                 size_t expert_packed_stride,
                 const dot_kernel& kernel) {
  assert(kernel.kernel != nullptr);

  size_t tile_k = kernel.tile_k;
  size_t block_m = kernel.block_m;

  // Stride of B in bytes to move by tile_k elements in reduction dimension.
  // packed_b_stride is input_b.dim(2).stride() in Slinky, which is the stride
  // of tiles_k. The kernel expects stride in bytes divided by tile_k.
  size_t b_stride_k1 = packed_b_stride / tile_k;

  // Stride of A (input_a) in bytes between rows (tokens).
  size_t a_stride_m = D_in * sizeof(float);

  // Stride of C (output) in bytes between rows (tokens).
  size_t c_stride_m = D_out * sizeof(float);

  // D_in must be a multiple of tile_k for the optimized kernel.
  assert(D_in % tile_k == 0);

  for (size_t e = 0; e < E; ++e) {
    int32_t count = expert_counts[e];
    if (count == 0) continue;

    int32_t offset = offsets[e];

    // Pointers for this expert.
    const float* A_e = input_a + offset * D_in;
    const void* B_e = offset_bytes(input_b, e * expert_packed_stride);
    float* C_e = output + offset * D_out;

    size_t M_e = static_cast<size_t>(count);

    // Loop over M in steps of block_m.
    for (size_t m_offset = 0; m_offset < M_e; m_offset += block_m) {
      size_t current_m = std::min(M_e - m_offset, block_m);

      const float* A_step = A_e + m_offset * D_in;
      float* C_step = C_e + m_offset * D_out;

      // Call the dot kernel.
      // k3=1, k2=1, k1=D_in.
      // a_stride_k3=0, a_stride_k2=0.
      // b_stride_k3=0, b_stride_k2=0.
      // c_in = nullptr (we don't accumulate, we just write).
      kernel.kernel(current_m, D_out, 1, 1, D_in, a_stride_m, 0, 0, A_step, 0,
                    0, b_stride_k1, B_e, 0, nullptr, c_stride_m, C_step);
    }
  }
}

}  // namespace ynn
