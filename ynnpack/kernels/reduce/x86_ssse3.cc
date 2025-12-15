// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/base/simd/x86_sse2.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace simd {

static s32x4 reduce_add(
    s32x4 a, s8x16 b, Identity /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  __m128i b2x = _mm_maddubs_epi16(_mm_set1_epi8(1), b.v);
  return a += s32x4{_mm_madd_epi16(_mm_set1_epi16(1), b2x)};
}

static s32x4 reduce_add(
    s32x4 a, u8x16 b, Identity /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  __m128i b2x = _mm_maddubs_epi16(b.v, _mm_set1_epi8(1));
  return a += s32x4{_mm_madd_epi16(_mm_set1_epi16(1), b2x)};
}

}  // namespace simd

using simd::s32x16;
using simd::s32x4;
using simd::s8x16;
using simd::u8x16;

void sum_int8_int32_ssse3(size_t n, size_t k3, size_t k2, size_t k1,
                          size_t a_stride_n, size_t a_stride_k3,
                          size_t a_stride_k2, const void* a, size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(int8_t)) {
    tiled_reduce<sum_accumulator_k1_1<s8x16, s32x16>, int8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const int8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x4, 16>, int8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const int8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

void sum_uint8_int32_ssse3(size_t n, size_t k3, size_t k2, size_t k1,
                           size_t a_stride_n, size_t a_stride_k3,
                           size_t a_stride_k2, const void* a, size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(uint8_t)) {
    tiled_reduce<sum_accumulator_k1_1<u8x16, s32x16>, uint8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x4, 16>, uint8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

}  // namespace ynn
