// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/transpose/interleave.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/log.h"
#include "ynnpack/base/type.h"

namespace ynn {

namespace {

template <typename T>
void interleave_impl(size_t factor, size_t m, size_t n, size_t stride_a,
                     const T* a, T* x) {
  if (factor == 1) {
    assert(m == 1);
    memcpy(x, a, n * sizeof(T));
    return;
  }
  assert(m <= factor);
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < m; ++i) {
      memcpy(x, offset_bytes(a, i * stride_a + j * sizeof(T)), sizeof(T));
      x = offset_bytes(x, sizeof(T));
    }
    for (size_t i = m; i < factor; ++i) {
      memset(x, 0, sizeof(T));
      x = offset_bytes(x, sizeof(T));
    }
  }
}

void interleave_impl(size_t factor, size_t m, size_t n, size_t stride_a,
                     const uint4x2* a, uint4x2* x) {
  assert(m <= factor);
  if (factor == 1) {
    memcpy(x, a, n / 2);
    return;
  }
  assert(n % 2 == 0);
  const size_t x_stride_n = factor / 2;
  for (size_t i = 0; i < factor / 2; ++i) {
    const int i0 = 2 * i + 0;
    const int i1 = 2 * i + 1;
    for (size_t j = 0; j < n / 2; ++j) {
      const uint4x2 a0 = i0 < m ? a[i0 * stride_a + j] : uint4x2(0, 0);
      const uint4x2 a1 = i1 < m ? a[i1 * stride_a + j] : uint4x2(0, 0);
      x[(2 * j + 0) * x_stride_n + i] = uint4x2(a0.get(0), a1.get(0));
      x[(2 * j + 1) * x_stride_n + i] = uint4x2(a0.get(1), a1.get(1));
    }
  }
}

}  // namespace

void interleave_x4(size_t factor, size_t m, size_t n, size_t stride_a,
                   const void* a, void* x) {
  interleave_impl(factor, m, n, stride_a, static_cast<const uint4x2*>(a),
                  static_cast<uint4x2*>(x));
}
void interleave_x8(size_t factor, size_t m, size_t n, size_t stride_a,
                   const void* a, void* x) {
  interleave_impl(factor, m, n, stride_a, static_cast<const uint8_t*>(a),
                  static_cast<uint8_t*>(x));
}
void interleave_x16(size_t factor, size_t m, size_t n, size_t stride_a,
                    const void* a, void* x) {
  interleave_impl(factor, m, n, stride_a, static_cast<const uint16_t*>(a),
                  static_cast<uint16_t*>(x));
}
void interleave_x32(size_t factor, size_t m, size_t n, size_t stride_a,
                    const void* a, void* x) {
  interleave_impl(factor, m, n, stride_a, static_cast<const uint32_t*>(a),
                  static_cast<uint32_t*>(x));
}
void interleave_x64(size_t factor, size_t m, size_t n, size_t stride_a,
                    const void* a, void* x) {
  interleave_impl(factor, m, n, stride_a, static_cast<const uint64_t*>(a),
                  static_cast<uint64_t*>(x));
}

interleave_kernel_fn get_interleave_kernel(size_t element_size_bits, size_t m) {
#define YNN_INTERLEAVE_KERNEL(arch_flags, name, M, type)                 \
  if (elem_size_of(type{}) == element_size_bits && (m == M || M == 0)) { \
    if (is_arch_supported(arch_flags)) {                                 \
      return name;                                                       \
    }                                                                    \
  }
#include "ynnpack/kernels/transpose/interleave.inc"
#undef YNN_INTERLEAVE_KERNEL
  YNN_LOG_ERROR() << "Unsupported interleave for " << m << " rows of "
                  << element_size_bits << "-bit elements.";
  return nullptr;
}

}  // namespace ynn
