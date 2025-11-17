// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstddef>
#include <type_traits>

#include "ynnpack/kernels/transpose/generic.h"
#include "ynnpack/kernels/transpose/transpose.h"

namespace ynn {

void transpose_x256_avx(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                        const void* a, size_t stride_x, void* x) {
  transpose(m, n, n_bytes_a, stride_a, a, stride_x, x,
            std::integral_constant<size_t, 256>{});
}
void transpose_x512_avx(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                        const void* a, size_t stride_x, void* x) {
  transpose(m, n, n_bytes_a, stride_a, a, stride_x, x,
            std::integral_constant<size_t, 512>{});
}
void transpose_x1024_avx(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                         const void* a, size_t stride_x, void* x) {
  transpose(m, n, n_bytes_a, stride_a, a, stride_x, x,
            std::integral_constant<size_t, 1024>{});
}
void transpose_x2048_avx(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                         const void* a, size_t stride_x, void* x) {
  transpose(m, n, n_bytes_a, stride_a, a, stride_x, x,
            std::integral_constant<size_t, 2048>{});
}

}  // namespace ynn
