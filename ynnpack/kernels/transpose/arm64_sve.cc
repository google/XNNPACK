// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/arm64_sve.h"

#include <array>
#include <cstddef>
#include <cstring>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/kernels/transpose/generic.h"
#include "ynnpack/kernels/transpose/interleave.h"
#include "ynnpack/kernels/transpose/transpose.h"

namespace ynn {

using simd::u8x8;
using simd::u8x16;

void transpose_x4_sve(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                      const void* a, size_t stride_x, void* x) {
  transpose<std::array<u8x8, 16>>(m, n, n_bytes_a, stride_a, a, stride_x, x,
                                  std::integral_constant<size_t, 4>{});
}
void transpose_x8_sve(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                      const void* a, size_t stride_x, void* x) {
  transpose<std::array<u8x16, 16>>(m, n, n_bytes_a, stride_a, a, stride_x, x,
                                   std::integral_constant<size_t, 8>{});
}
void transpose_x16_sve(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                       const void* a, size_t stride_x, void* x) {
  transpose<std::array<u8x16, 8>>(m, n, n_bytes_a, stride_a, a, stride_x, x,
                                  std::integral_constant<size_t, 16>{});
}
void transpose_x32_sve(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                       const void* a, size_t stride_x, void* x) {
  transpose<std::array<u8x16, 4>>(m, n, n_bytes_a, stride_a, a, stride_x, x,
                                  std::integral_constant<size_t, 32>{});
}
void transpose_x64_sve(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                       const void* a, size_t stride_x, void* x) {
  transpose<std::array<u8x16, 2>>(m, n, n_bytes_a, stride_a, a, stride_x, x,
                                  std::integral_constant<size_t, 64>{});
}

void interleave2_x4_sve(size_t factor, size_t m, size_t n, size_t stride_a,
                        const void* a, void* x) {
  assert(factor == 2);
  interleave<std::array<u8x16, 2>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 4>{});
}
void interleave2_x8_sve(size_t factor, size_t m, size_t n, size_t stride_a,
                        const void* a, void* x) {
  assert(factor == 2);
  interleave<std::array<u8x16, 2>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 8>{});
}
void interleave2_x16_sve(size_t factor, size_t m, size_t n, size_t stride_a,
                         const void* a, void* x) {
  assert(factor == 2);
  interleave<std::array<u8x16, 2>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 16>{});
}
void interleave2_x32_sve(size_t factor, size_t m, size_t n, size_t stride_a,
                         const void* a, void* x) {
  assert(factor == 2);
  interleave<std::array<u8x16, 2>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 32>{});
}
void interleave4_x8_sve(size_t factor, size_t m, size_t n, size_t stride_a,
                        const void* a, void* x) {
  assert(factor == 4);
  interleave<std::array<u8x16, 4>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 8>{});
}
void interleave4_x16_sve(size_t factor, size_t m, size_t n, size_t stride_a,
                         const void* a, void* x) {
  assert(factor == 4);
  interleave<std::array<u8x16, 4>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 16>{});
}
void interleave4_x32_sve(size_t factor, size_t m, size_t n, size_t stride_a,
                         const void* a, void* x) {
  assert(factor == 4);
  interleave<std::array<u8x16, 4>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 32>{});
}

}  // namespace ynn
