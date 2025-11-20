// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/transpose/hvx.h"

#include <hexagon_types.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>

#include "ynnpack/kernels/transpose/generic.h"
#include "ynnpack/kernels/transpose/interleave.h"
#include "ynnpack/kernels/transpose/transpose.h"

namespace ynn {

void transpose_x32_hvx(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                       const void* a, size_t stride_x, void* x) {
  transpose<std::array<HVX_Vector, 32>>(m, n, n_bytes_a, stride_a, a, stride_x,
                                        x,
                                        std::integral_constant<size_t, 32>{});
}
void transpose_x64_hvx(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                       const void* a, size_t stride_x, void* x) {
  transpose<std::array<HVX_Vector, 16>>(m, n, n_bytes_a, stride_a, a, stride_x,
                                        x,
                                        std::integral_constant<size_t, 64>{});
}
void transpose_x128_hvx(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                        const void* a, size_t stride_x, void* x) {
  transpose<std::array<HVX_Vector, 8>>(m, n, n_bytes_a, stride_a, a, stride_x,
                                       x,
                                       std::integral_constant<size_t, 128>{});
}
void transpose_x256_hvx(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                        const void* a, size_t stride_x, void* x) {
  transpose<std::array<HVX_Vector, 4>>(m, n, n_bytes_a, stride_a, a, stride_x,
                                       x,
                                       std::integral_constant<size_t, 256>{});
}
void transpose_x512_hvx(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                        const void* a, size_t stride_x, void* x) {
  transpose<std::array<HVX_Vector, 2>>(m, n, n_bytes_a, stride_a, a, stride_x,
                                       x,
                                       std::integral_constant<size_t, 512>{});
}
void transpose_x1024_hvx(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                         const void* a, size_t stride_x, void* x) {
  transpose(m, n, n_bytes_a, stride_a, a, stride_x, x,
            std::integral_constant<size_t, 1024>{});
}

void interleave2_x8_hvx(size_t factor, size_t m, size_t n, size_t stride_a,
                        const void* a, void* x) {
  assert(factor == 2);
  interleave<std::array<HVX_Vector, 2>>(m, n, stride_a, a, x,
                                        std::integral_constant<size_t, 8>{});
}

void interleave2_x16_hvx(size_t factor, size_t m, size_t n, size_t stride_a,
                         const void* a, void* x) {
  assert(factor == 2);
  interleave<std::array<HVX_Vector, 2>>(m, n, stride_a, a, x,
                                        std::integral_constant<size_t, 16>{});
}

void interleave2_x32_hvx(size_t factor, size_t m, size_t n, size_t stride_a,
                         const void* a, void* x) {
  assert(factor == 2);
  interleave<std::array<HVX_Vector, 2>>(m, n, stride_a, a, x,
                                        std::integral_constant<size_t, 32>{});
}

void interleave4_x8_hvx(size_t factor, size_t m, size_t n, size_t stride_a,
                        const void* a, void* x) {
  assert(factor == 4);
  interleave<std::array<HVX_Vector, 4>>(m, n, stride_a, a, x,
                                        std::integral_constant<size_t, 8>{});
}

void interleave4_x16_hvx(size_t factor, size_t m, size_t n, size_t stride_a,
                         const void* a, void* x) {
  assert(factor == 4);
  interleave<std::array<HVX_Vector, 4>>(m, n, stride_a, a, x,
                                        std::integral_constant<size_t, 16>{});
}

void interleave4_x32_hvx(size_t factor, size_t m, size_t n, size_t stride_a,
                         const void* a, void* x) {
  assert(factor == 4);
  interleave<std::array<HVX_Vector, 4>>(m, n, stride_a, a, x,
                                        std::integral_constant<size_t, 32>{});
}

}  // namespace ynn
