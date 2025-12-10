// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_sse41.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/base/simd/multi_vec.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max_accumulator.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace simd {

using s32x4x2 = multi_vec<s32x4, 2>;
using s32x4x4 = multi_vec<s32x4, 4>;

s32x4x4& operator+=(s32x4x4& a, s8x16 b) {
  s32x4 b_0(_mm_cvtepi8_epi32(b.v));
  s32x4 b_1(_mm_cvtepi8_epi32(_mm_srli_si128(b.v, 4)));
  s32x4 b_2(_mm_cvtepi8_epi32(_mm_srli_si128(b.v, 8)));
  s32x4 b_3(_mm_cvtepi8_epi32(_mm_srli_si128(b.v, 12)));

  a.v[0] += b_0;
  a.v[1] += b_1;
  a.v[2] += b_2;
  a.v[3] += b_3;
  return a;
}

s32x4x4& operator+=(s32x4x4& a, u8x16 b) {
  s32x4 b_0(_mm_cvtepu8_epi32(b.v));
  s32x4 b_1(_mm_cvtepu8_epi32(_mm_srli_si128(b.v, 4)));
  s32x4 b_2(_mm_cvtepu8_epi32(_mm_srli_si128(b.v, 8)));
  s32x4 b_3(_mm_cvtepu8_epi32(_mm_srli_si128(b.v, 12)));

  a.v[0] += b_0;
  a.v[1] += b_1;
  a.v[2] += b_2;
  a.v[3] += b_3;
  return a;
}

static s32x4x4 reduce_add(
    s32x4x4 a, s8x16 b, Square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  // Convert int8 -> uint8 via abs first.
  __m128i abs_b = _mm_abs_epi8(b.v);

  __m128i b_0 = _mm_cvtepu8_epi32(abs_b);
  __m128i b_1 = _mm_cvtepu8_epi32(_mm_srli_si128(abs_b, 4));
  __m128i b_2 = _mm_cvtepu8_epi32(_mm_srli_si128(abs_b, 8));
  __m128i b_3 = _mm_cvtepu8_epi32(_mm_srli_si128(abs_b, 12));

  // madd_epi16 works due to extra zeros from uint8 -> int32 conversion.
  a.v[0] += s32x4{_mm_madd_epi16(b_0, b_0)};
  a.v[1] += s32x4{_mm_madd_epi16(b_1, b_1)};
  a.v[2] += s32x4{_mm_madd_epi16(b_2, b_2)};
  a.v[3] += s32x4{_mm_madd_epi16(b_3, b_3)};

  return a;
}

static s32x4x4 reduce_add(
    s32x4x4 a, u8x16 b, Square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  __m128i b_0 = _mm_cvtepu8_epi32(b.v);
  __m128i b_1 = _mm_cvtepu8_epi32(_mm_srli_si128(b.v, 4));
  __m128i b_2 = _mm_cvtepu8_epi32(_mm_srli_si128(b.v, 8));
  __m128i b_3 = _mm_cvtepu8_epi32(_mm_srli_si128(b.v, 12));

  // madd_epi16 works due to extra zeros from uint8 -> int32 conversion.
  a.v[0] += s32x4{_mm_madd_epi16(b_0, b_0)};
  a.v[1] += s32x4{_mm_madd_epi16(b_1, b_1)};
  a.v[2] += s32x4{_mm_madd_epi16(b_2, b_2)};
  a.v[3] += s32x4{_mm_madd_epi16(b_3, b_3)};

  return a;
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

using simd::s32x4;
using simd::s32x4x2;
using simd::s32x4x4;
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
    tiled_reduce<sum_accumulator_k1_1<s8x16, s32x4x4>, int8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const int8_t*>(a),
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
    tiled_reduce<sum_accumulator_k1_1<u8x16, s32x4x4>, uint8_t, int32_t>(
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
    tiled_reduce<sum_accumulator_k1_1<s8x16, s32x4x4, Square>, int8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const int8_t*>(a),
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
    tiled_reduce<sum_accumulator_k1_1<u8x16, s32x4x4, Square>, uint8_t,
      int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x4, 16, Square>, uint8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

}  // namespace ynn
