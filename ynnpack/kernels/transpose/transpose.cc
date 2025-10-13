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

#include "ynnpack/base/arch.h"
#include "ynnpack/base/arithmetic.h"
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
  transpose(m, n, n_bytes_a, stride_a, static_cast<const uint8_t*>(a), stride_x,
            static_cast<uint8_t*>(x));
}
void transpose_x16(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                   const void* a, size_t stride_x, void* x) {
  transpose(m, n, n_bytes_a, stride_a, static_cast<const uint16_t*>(a),
            stride_x, static_cast<uint16_t*>(x));
}
void transpose_x32(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                   const void* a, size_t stride_x, void* x) {
  transpose(m, n, n_bytes_a, stride_a, static_cast<const uint32_t*>(a),
            stride_x, static_cast<uint32_t*>(x));
}
void transpose_x64(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                   const void* a, size_t stride_x, void* x) {
  transpose(m, n, n_bytes_a, stride_a, static_cast<const uint64_t*>(a),
            stride_x, static_cast<uint64_t*>(x));
}

transpose_kernel_fn get_transpose_kernel(size_t element_size_bits) {
#define YNN_TRANSPOSE_KERNEL(arch_flags, name, type)         \
  if (sizeof(type) * 8 / type_info<type>::element_count() == \
      element_size_bits) {                                   \
    if (is_arch_supported(arch_flags)) {                     \
      return name;                                           \
    }                                                        \
  }
#include "ynnpack/kernels/transpose/transpose.inc"
#undef YNN_INTERLEAVE_KERNEL
  YNN_LOG_DEBUG() << "Unsupported transpose of " << element_size_bits
                  << "-bit elements.";
  return nullptr;
}

}  // namespace ynn
