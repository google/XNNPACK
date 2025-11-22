// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/transpose/transpose.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/log.h"
#include "ynnpack/base/type.h"
#include "ynnpack/kernels/transpose/generic.h"

namespace ynn {

namespace {

void transpose(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
               const uint4x2* a, size_t stride_x, uint4x2* x) {
  assert(m % 2 == 0);
  assert(n % 2 == 0);
  // Handle the in bounds columns first.
  const size_t n_bytes = std::min(m / 2, n_bytes_a);
  for (size_t j = 0; j < n / 2; ++j) {
    for (size_t i = 0; i < n_bytes; ++i) {
      const uint4x2 a0 = a[(2 * j + 0) * stride_a + i];
      const uint4x2 a1 = a[(2 * j + 1) * stride_a + i];
      x[(2 * i + 0) * stride_x + j] = uint4x2(a0.get(0), a1.get(0));
      x[(2 * i + 1) * stride_x + j] = uint4x2(a0.get(1), a1.get(1));
    }
  }
  // Handle any out of bounds columns of input (rows of output).
  for (size_t i = n_bytes * 2; i < m; ++i) {
    memset(&x[i * stride_x], 0, n / 2);
  }
}

}  // namespace

void transpose_x4(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                  const void* a, size_t stride_x, void* x) {
  transpose(m, n, n_bytes_a, stride_a, static_cast<const uint4x2*>(a), stride_x,
            static_cast<uint4x2*>(x));
}
void transpose_x8(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                  const void* a, size_t stride_x, void* x) {
  transpose(m, n, n_bytes_a, stride_a, a, stride_x, x,
            std::integral_constant<size_t, 8>{});
}
void transpose_x16(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                   const void* a, size_t stride_x, void* x) {
  transpose(m, n, n_bytes_a, stride_a, a, stride_x, x,
            std::integral_constant<size_t, 16>{});
}
void transpose_x32(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                   const void* a, size_t stride_x, void* x) {
  transpose(m, n, n_bytes_a, stride_a, a, stride_x, x,
            std::integral_constant<size_t, 32>{});
}
void transpose_x64(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                   const void* a, size_t stride_x, void* x) {
  transpose(m, n, n_bytes_a, stride_a, a, stride_x, x,
            std::integral_constant<size_t, 64>{});
}
void transpose_x128(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                    const void* a, size_t stride_x, void* x) {
  transpose(m, n, n_bytes_a, stride_a, a, stride_x, x,
            std::integral_constant<size_t, 128>{});
}

transpose_kernel_fn get_transpose_kernel(size_t element_size_bits) {
#define YNN_TRANSPOSE_KERNEL(arch_flags, name, kernel_element_size_bits) \
  if (kernel_element_size_bits == element_size_bits) {                   \
    if (is_arch_supported(arch_flags)) {                                 \
      return name;                                                       \
    }                                                                    \
  }
#include "ynnpack/kernels/transpose/transpose.inc"
#undef YNN_TRANSPOSE_KERNEL
  YNN_LOG_DEBUG() << "Unsupported transpose of " << element_size_bits
                  << "-bit elements.";
  return nullptr;
}

namespace {

void tile_transpose(size_t tile, size_t elem_size_bits, size_t m, size_t n,
                    size_t n_bytes_a, size_t stride_a, const void* a,
                    size_t stride_x, void* x,
                    transpose_kernel_fn transpose_fn) {
  assert(transpose_fn);
  assert(tile * elem_size_bits % 8 == 0);
  // Our transpose kernels loop over rows, then columns, so we only need to
  // tile the column dimension to get good memory locality for both reads and
  // writes.
  while (n > 0) {
    transpose_fn(m, std::min(n, tile), n_bytes_a, stride_a, a, stride_x, x);
    n = sub_sat(n, tile);
    x = offset_bytes(x, tile * elem_size_bits / 8);
    a = offset_bytes(a, tile * stride_a);
  }
}

}  // namespace

transpose_fn make_tiled_transpose(size_t elem_size_bits,
                                  transpose_kernel_fn transpose_fn) {
  assert(transpose_fn);
  constexpr size_t tile_size_bits = YNN_CACHE_LINE_SIZE * 8;
  const size_t tile = std::max<size_t>(16, tile_size_bits / elem_size_bits);
  return [elem_size_bits, transpose_fn, tile](
             size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
             const void* a, size_t stride_x, void* x) {
    tile_transpose(tile, elem_size_bits, m, n, n_bytes_a, stride_a, a, stride_x,
                   x, transpose_fn);
  };
}

transpose_fn get_tiled_transpose(size_t elem_size_bits) {
  transpose_kernel_fn transpose_fn = get_transpose_kernel(elem_size_bits);
  if (!transpose_fn) return nullptr;

  return make_tiled_transpose(elem_size_bits, transpose_fn);
}

}  // namespace ynn
