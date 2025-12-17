// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_sse41.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max_accumulator.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace simd {

static s32x16 reduce_add(
    s32x16 a, u8x16 b, Square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  s32x16 b_s32 = convert(b, int32_t{});

  // madd_epi16 works due to extra zeros from uint8 -> int32 conversion.
  a[0] += s32x4{_mm_madd_epi16(b_s32[0].v, b_s32[0].v)};
  a[1] += s32x4{_mm_madd_epi16(b_s32[1].v, b_s32[1].v)};
  a[2] += s32x4{_mm_madd_epi16(b_s32[2].v, b_s32[2].v)};
  a[3] += s32x4{_mm_madd_epi16(b_s32[3].v, b_s32[3].v)};

  return a;
}

static s32x16 reduce_add(s32x16 a, s8x16 b, Square map_fn,
                         std::integral_constant<size_t, 1> horizontal_factor) {
  // We're squaring, we can take the absolute value and use the unsigned reduce.
  return reduce_add(a, abs(b), map_fn, horizontal_factor);
}

static s32x4 reduce_add(
    s32x4 a, s8x16 b, Square /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  __m128i lo = _mm_cvtepi8_epi16(b.v);
  __m128i hi = _mm_cvtepi8_epi16(_mm_srli_si128(b.v, 8));
  return a += s32x4(_mm_hadd_epi32(_mm_madd_epi16(lo, lo),
                                   _mm_madd_epi16(hi, hi)));
}

static s32x4 reduce_add(
    s32x4 a, u8x16 b, Square /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  __m128i lo = _mm_cvtepu8_epi16(b.v);
  __m128i hi = _mm_cvtepu8_epi16(_mm_srli_si128(b.v, 8));
  return a += s32x4(_mm_hadd_epi32(_mm_madd_epi16(lo, lo),
                                   _mm_madd_epi16(hi, hi)));
}

}  // namespace simd

using simd::s32x16;
using simd::s32x4;
using simd::s8x16;
using simd::u8x16;

MIN_MAX_KERNEL(min_max_int8_4x16_sse41, s8x16, s8x16, int8_t, 16);
MIN_MAX_KERNEL(min_int8_4x16_sse41, s8x16, dummy_t, int8_t, 16);
MIN_MAX_KERNEL(max_int8_4x16_sse41, dummy_t, s8x16, int8_t, 16);

void sum_int8_int32_sse2(size_t n, size_t k3, size_t k2, size_t k1,
                         size_t a_stride_n, size_t a_stride_k3,
                         size_t a_stride_k2, const void* a, size_t, void* c);
void sum_uint8_int32_sse2(size_t n, size_t k3, size_t k2, size_t k1,
                          size_t a_stride_n, size_t a_stride_k3,
                          size_t a_stride_k2, const void* a, size_t, void* c);

void sum_int8_int32_sse41(size_t n, size_t k3, size_t k2, size_t k1,
                          size_t a_stride_n, size_t a_stride_k3,
                          size_t a_stride_k2, const void* a, size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(int8_t)) {
    stream_reduce<sum_accumulator_k1_1<s8x16, s32x16>, int8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const int8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    sum_int8_int32_sse2(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2, a, 0, c);
  }
}

void sum_uint8_int32_sse41(size_t n, size_t k3, size_t k2, size_t k1,
                           size_t a_stride_n, size_t a_stride_k3,
                           size_t a_stride_k2, const void* a, size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(uint8_t)) {
    stream_reduce<sum_accumulator_k1_1<u8x16, s32x16>, uint8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
      sum_uint8_int32_sse2(
          n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2, a, 0, c);
  }
}

void sum_squared_int8_int32_sse41(size_t n, size_t k3, size_t k2, size_t k1,
                                  size_t a_stride_n, size_t a_stride_k3,
                                  size_t a_stride_k2, const void* a, size_t,
                                  void* c) {
  if (k1 == 1 && a_stride_n == sizeof(int8_t)) {
    stream_reduce<sum_accumulator_k1_1<s8x16, s32x16, Square>, int8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const int8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x4, 16, Square>, int8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const int8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

void sum_squared_uint8_int32_sse41(size_t n, size_t k3, size_t k2, size_t k1,
                                   size_t a_stride_n, size_t a_stride_k3,
                                   size_t a_stride_k2, const void* a, size_t,
                                   void* c) {
  if (k1 == 1 && a_stride_n == sizeof(uint8_t)) {
    stream_reduce<sum_accumulator_k1_1<u8x16, s32x16, Square>, uint8_t,
                  int32_t>(n, k3, k2, a_stride_k3, a_stride_k2,
                           reinterpret_cast<const uint8_t*>(a),
                           /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x4, 16, Square>, uint8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

}  // namespace ynn
