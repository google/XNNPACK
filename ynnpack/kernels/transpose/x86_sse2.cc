// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/transpose/x86_sse2.h"

#include <immintrin.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "ynnpack/base/type.h"
#include "ynnpack/kernels/transpose/generic.h"
#include "ynnpack/kernels/transpose/interleave.h"
#include "ynnpack/kernels/transpose/transpose.h"

namespace ynn {

void transpose_x4_sse2(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                       const void* a, size_t stride_x, void* x) {
  transpose<std::array<__m128i, 32>>(m, n, n_bytes_a, stride_a,
                                     static_cast<const uint4x2*>(a), stride_x,
                                     static_cast<uint4x2*>(x));
}
void transpose_x8_sse2(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                       const void* a, size_t stride_x, void* x) {
  transpose<std::array<__m128i, 16>>(m, n, n_bytes_a, stride_a,
                                     static_cast<const uint8_t*>(a), stride_x,
                                     static_cast<uint8_t*>(x));
}
void transpose_x16_sse2(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                        const void* a, size_t stride_x, void* x) {
  transpose<std::array<__m128i, 8>>(m, n, n_bytes_a, stride_a,
                                    static_cast<const uint16_t*>(a), stride_x,
                                    static_cast<uint16_t*>(x));
}
void transpose_x32_sse2(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                        const void* a, size_t stride_x, void* x) {
  transpose<std::array<__m128i, 4>>(m, n, n_bytes_a, stride_a,
                                    static_cast<const uint32_t*>(a), stride_x,
                                    static_cast<uint32_t*>(x));
}
void transpose_x64_sse2(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                        const void* a, size_t stride_x, void* x) {
  transpose<std::array<__m128i, 2>>(m, n, n_bytes_a, stride_a,
                                    static_cast<const uint64_t*>(a), stride_x,
                                    static_cast<uint64_t*>(x));
}
void transpose_x128_sse2(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                         const void* a, size_t stride_x, void* x) {
  transpose(m, n, n_bytes_a, stride_a, static_cast<const x128_t*>(a), stride_x,
            static_cast<x128_t*>(x));
}
void transpose_x256_sse2(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                         const void* a, size_t stride_x, void* x) {
  transpose(m, n, n_bytes_a, stride_a, static_cast<const x256_t*>(a), stride_x,
            static_cast<x256_t*>(x));
}
void transpose_x512_sse2(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                         const void* a, size_t stride_x, void* x) {
  transpose(m, n, n_bytes_a, stride_a, static_cast<const x512_t*>(a), stride_x,
            static_cast<x512_t*>(x));
}
void transpose_x1024_sse2(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                          const void* a, size_t stride_x, void* x) {
  transpose(m, n, n_bytes_a, stride_a, static_cast<const x1024_t*>(a), stride_x,
            static_cast<x1024_t*>(x));
}

void interleave2_x4_sse2(size_t factor, size_t m, size_t n, size_t stride_a,
                         const void* a, void* x) {
  assert(factor == 2);
  interleave<std::array<__m128i, 2>>(
      m, n, stride_a, static_cast<const uint4x2*>(a), static_cast<uint4x2*>(x));
}

void interleave2_x8_sse2(size_t factor, size_t m, size_t n, size_t stride_a,
                         const void* a, void* x) {
  assert(factor == 2);
  interleave<std::array<__m128i, 2>>(
      m, n, stride_a, static_cast<const uint8_t*>(a), static_cast<uint8_t*>(x));
}

void interleave2_x16_sse2(size_t factor, size_t m, size_t n, size_t stride_a,
                          const void* a, void* x) {
  assert(factor == 2);
  interleave<std::array<__m128i, 2>>(m, n, stride_a,
                                     static_cast<const uint16_t*>(a),
                                     static_cast<uint16_t*>(x));
}

void interleave2_x32_sse2(size_t factor, size_t m, size_t n, size_t stride_a,
                          const void* a, void* x) {
  assert(factor == 2);
  interleave<std::array<__m128i, 2>>(m, n, stride_a,
                                     static_cast<const uint32_t*>(a),
                                     static_cast<uint32_t*>(x));
}

void interleave4_x8_sse2(size_t factor, size_t m, size_t n, size_t stride_a,
                         const void* a, void* x) {
  assert(factor == 4);
  interleave<std::array<__m128i, 4>>(
      m, n, stride_a, static_cast<const uint8_t*>(a), static_cast<uint8_t*>(x));
}

void interleave4_x16_sse2(size_t factor, size_t m, size_t n, size_t stride_a,
                          const void* a, void* x) {
  assert(factor == 4);
  interleave<std::array<__m128i, 4>>(m, n, stride_a,
                                     static_cast<const uint16_t*>(a),
                                     static_cast<uint16_t*>(x));
}

void interleave4_x32_sse2(size_t factor, size_t m, size_t n, size_t stride_a,
                          const void* a, void* x) {
  assert(factor == 4);
  interleave<std::array<__m128i, 4>>(m, n, stride_a,
                                     static_cast<const uint32_t*>(a),
                                     static_cast<uint32_t*>(x));
}

}  // namespace ynn
