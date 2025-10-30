// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/transpose/arm_neon.h"

#include <arm_neon.h>

#include <array>
#include <cstddef>

#include "ynnpack/kernels/transpose/generic.h"
#include "ynnpack/kernels/transpose/transpose.h"

namespace ynn {

void transpose_x4_neon(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                       const void* a, size_t stride_x, void* x) {
  transpose<std::array<uint8x8_t, 16>>(m, n, n_bytes_a, stride_a,
                                       static_cast<const uint4x2*>(a), stride_x,
                                       static_cast<uint4x2*>(x));
}
void transpose_x8_neon(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                       const void* a, size_t stride_x, void* x) {
  transpose<std::array<uint8x16_t, 16>>(m, n, n_bytes_a, stride_a,
                                        static_cast<const uint8_t*>(a),
                                        stride_x, static_cast<uint8_t*>(x));
}
void transpose_x16_neon(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                        const void* a, size_t stride_x, void* x) {
  transpose<std::array<uint8x16_t, 8>>(m, n, n_bytes_a, stride_a,
                                       static_cast<const uint16_t*>(a),
                                       stride_x, static_cast<uint16_t*>(x));
}
void transpose_x32_neon(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                        const void* a, size_t stride_x, void* x) {
  transpose<std::array<uint8x16_t, 4>>(m, n, n_bytes_a, stride_a,
                                       static_cast<const uint32_t*>(a),
                                       stride_x, static_cast<uint32_t*>(x));
}
void transpose_x64_neon(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                        const void* a, size_t stride_x, void* x) {
  transpose<std::array<uint8x16_t, 2>>(m, n, n_bytes_a, stride_a,
                                       static_cast<const uint64_t*>(a),
                                       stride_x, static_cast<uint64_t*>(x));
}
void transpose_x128_neon(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                         const void* a, size_t stride_x, void* x) {
  transpose(m, n, n_bytes_a, stride_a, static_cast<const x128_t*>(a), stride_x,
            static_cast<x128_t*>(x));
}
void transpose_x256_neon(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                         const void* a, size_t stride_x, void* x) {
  transpose(m, n, n_bytes_a, stride_a, static_cast<const x256_t*>(a), stride_x,
            static_cast<x256_t*>(x));
}
void transpose_x512_neon(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                         const void* a, size_t stride_x, void* x) {
  transpose(m, n, n_bytes_a, stride_a, static_cast<const x512_t*>(a), stride_x,
            static_cast<x512_t*>(x));
}
void transpose_x1024_neon(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                          const void* a, size_t stride_x, void* x) {
  transpose(m, n, n_bytes_a, stride_a, static_cast<const x1024_t*>(a), stride_x,
            static_cast<x1024_t*>(x));
}

void interleave2_x4_neon(size_t factor, size_t m, size_t n, size_t stride_a,
                         const void* a, void* x) {
  assert(factor == 2);
  interleave<std::array<uint8x16_t, 2>>(
      m, n, stride_a, static_cast<const uint4x2*>(a), static_cast<uint4x2*>(x));
}
void interleave2_x8_neon(size_t factor, size_t m, size_t n, size_t stride_a,
                         const void* a, void* x) {
  assert(factor == 2);
  interleave<std::array<uint8x16_t, 2>>(
      m, n, stride_a, static_cast<const uint8_t*>(a), static_cast<uint8_t*>(x));
}
void interleave2_x16_neon(size_t factor, size_t m, size_t n, size_t stride_a,
                          const void* a, void* x) {
  assert(factor == 2);
  interleave<std::array<uint8x16_t, 2>>(m, n, stride_a,
                                        static_cast<const uint16_t*>(a),
                                        static_cast<uint16_t*>(x));
}
void interleave2_x32_neon(size_t factor, size_t m, size_t n, size_t stride_a,
                          const void* a, void* x) {
  assert(factor == 2);
  interleave<std::array<uint8x16_t, 2>>(m, n, stride_a,
                                        static_cast<const uint32_t*>(a),
                                        static_cast<uint32_t*>(x));
}
void interleave4_x8_neon(size_t factor, size_t m, size_t n, size_t stride_a,
                         const void* a, void* x) {
  assert(factor == 4);
  interleave<std::array<uint8x16_t, 4>>(
      m, n, stride_a, static_cast<const uint8_t*>(a), static_cast<uint8_t*>(x));
}
void interleave4_x16_neon(size_t factor, size_t m, size_t n, size_t stride_a,
                          const void* a, void* x) {
  assert(factor == 4);
  interleave<std::array<uint8x16_t, 4>>(m, n, stride_a,
                                        static_cast<const uint16_t*>(a),
                                        static_cast<uint16_t*>(x));
}
void interleave4_x32_neon(size_t factor, size_t m, size_t n, size_t stride_a,
                          const void* a, void* x) {
  assert(factor == 4);
  interleave<std::array<uint8x16_t, 4>>(m, n, stride_a,
                                        static_cast<const uint32_t*>(a),
                                        static_cast<uint32_t*>(x));
}

}  // namespace ynn
