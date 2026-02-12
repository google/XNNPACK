// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_avx2.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/kernels/transpose/interleave.h"
#include "ynnpack/kernels/transpose/transpose.h"

namespace ynn {

using simd::u8x32;

// TODO(b/450992581): This is faster than using simd::load (the default).
template <size_t M>
static std::array<u8x32, M> load(std::array<u8x32, M>, const void* a,
                                 size_t stride, size_t m, size_t n_bytes) {
  std::array<u8x32, M> result;
  memset(&result, 0, sizeof(result));
  for (size_t i = 0; i < m; ++i) {
    memcpy(&result[i], offset_bytes(a, i * stride), n_bytes);
  }
  return result;
}

}  // namespace ynn

#include "ynnpack/kernels/transpose/generic.h"

namespace ynn {

void transpose_x32_avx2(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                        const void* a, size_t stride_x, void* x) {
  transpose<std::array<u8x32, 8>>(m, n, n_bytes_a, stride_a, a, stride_x, x,
                                  std::integral_constant<size_t, 32>{});
}
void transpose_x64_avx2(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                        const void* a, size_t stride_x, void* x) {
  transpose<std::array<u8x32, 4>>(m, n, n_bytes_a, stride_a, a, stride_x, x,
                                  std::integral_constant<size_t, 64>{});
}

// The 128-bit avx2 kernel is faster than SSE2, *if* the transpose is not tiled.

void interleave2_x4_avx2(size_t factor, size_t m, size_t n, size_t stride_a,
                         const void* a, void* x) {
  assert(factor == 2);
  interleave<std::array<u8x32, 2>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 4>{});
}

void interleave2_x8_avx2(size_t factor, size_t m, size_t n, size_t stride_a,
                         const void* a, void* x) {
  assert(factor == 2);
  interleave<std::array<u8x32, 2>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 8>{});
}

void interleave2_x16_avx2(size_t factor, size_t m, size_t n, size_t stride_a,
                          const void* a, void* x) {
  assert(factor == 2);
  interleave<std::array<u8x32, 2>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 16>{});
}

void interleave2_x32_avx2(size_t factor, size_t m, size_t n, size_t stride_a,
                          const void* a, void* x) {
  assert(factor == 2);
  interleave<std::array<u8x32, 2>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 32>{});
}

void interleave4_x8_avx2(size_t factor, size_t m, size_t n, size_t stride_a,
                         const void* a, void* x) {
  assert(factor == 4);
  interleave<std::array<u8x32, 4>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 8>{});
}

void interleave4_x16_avx2(size_t factor, size_t m, size_t n, size_t stride_a,
                          const void* a, void* x) {
  assert(factor == 4);
  interleave<std::array<u8x32, 4>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 16>{});
}

void interleave4_x32_avx2(size_t factor, size_t m, size_t n, size_t stride_a,
                          const void* a, void* x) {
  assert(factor == 4);
  interleave<std::array<u8x32, 4>>(m, n, stride_a, a, x,
                                   std::integral_constant<size_t, 32>{});
}

}  // namespace ynn
