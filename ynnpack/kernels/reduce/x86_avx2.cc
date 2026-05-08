// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_avx2.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum.h"

namespace ynn {

namespace simd {

static s32x16 reduce_add(
    s32x16 a, u8x16 b, square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  __m256i b_lo = _mm256_cvtepu8_epi32(b.v);
  __m256i b_hi = _mm256_cvtepu8_epi32(_mm_bsrli_si128(b.v, 8));

  // madd_epi16 works due to extra zeros from uint8 -> int32 conversion.
  a[0] += s32x8{_mm256_madd_epi16(b_lo, b_lo)};
  a[1] += s32x8{_mm256_madd_epi16(b_hi, b_hi)};
  return a;
}

static s32x16 reduce_add(s32x16 a, s8x16 b, square map_fn,
                         std::integral_constant<size_t, 1> horizontal_factor) {
  // We're squaring, we can take the absolute value and use the unsigned reduce.
  return reduce_add(a, abs(b), map_fn, horizontal_factor);
}

static s32x8 reduce_add(
    s32x8 a, s8x32 b, identity /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  __m256i b2x = _mm256_maddubs_epi16(_mm256_set1_epi8(1), b.v);
  s32x8 b_s32(_mm256_madd_epi16(_mm256_set1_epi16(1), b2x));
  return a += b_s32;
}

static s32x8 reduce_add(
    s32x8 a, u8x32 b, identity /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  __m256i b2x = _mm256_maddubs_epi16(b.v, _mm256_set1_epi8(1));
  s32x8 b_s32(_mm256_madd_epi16(_mm256_set1_epi16(1), b2x));
  return a += b_s32;
}

static s32x8 reduce_add(
    s32x8 a, s8x16 b, square /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  __m256i b_16 = _mm256_cvtepi8_epi16(b.v);
  return a += s32x8(_mm256_madd_epi16(b_16, b_16));
}

static s32x8 reduce_add(
    s32x8 a, u8x16 b, square /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  __m256i b_16 = _mm256_cvtepu8_epi16(b.v);
  return a += s32x8(_mm256_madd_epi16(b_16, b_16));
}

template <typename MapFn>
static f32x8 reduce_add(
    f32x8 a, bf16x16 b, MapFn map_fn,
    std::integral_constant<size_t, 2> /*horizontal_factor*/ = {}) {
  __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0xFFFF0000));
  f32x8 evens(_mm256_castsi256_ps(_mm256_slli_epi32(b.v, 16)));
  f32x8 odds(_mm256_and_ps(_mm256_castsi256_ps(b.v), mask));

  a += map_fn(odds);
  a += map_fn(evens);
  return a;
}

using f32x16 = simd::vec<float, 16>;
using bf16x32 = simd::vec<bfloat16, 32>;

template <typename MapFn>
static f32x16 reduce_add(
    f32x16 a, bf16x32 b, MapFn map_fn,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  f32x8 a0 =
      reduce_add(extract<0>(a, f32x8::N), extract<0>(b, bf16x16::N), map_fn);
  f32x8 a1 =
      reduce_add(extract<1>(a, f32x8::N), extract<1>(b, bf16x16::N), map_fn);
  return {a0, a1};
}

}  // namespace simd

using simd::bf16x8;
using simd::bf16x16;
using simd::f16x16;
using simd::f32x8;
using simd::f32x16;
using simd::s16x16;
using simd::s32x16;
using simd::s32x32;
using simd::s32x8;
using simd::s8x16;
using simd::s8x32;
using simd::u8x16;
using simd::u8x32;

using f16x16_rvar = float16_wrapper<f16x16, s16x16>;
using bf16x16_rvar = float16_wrapper<bf16x16, s16x16>;

MIN_MAX_K1_KERNEL(min_max_bf16_k1_avx2, bf16x16_rvar, bf16x16_rvar, bfloat16,
                  16);
MIN_MAX_KN_KERNEL(min_max_bf16_kn_avx2, bf16x16_rvar, bf16x16_rvar, bfloat16,
                  16);
MIN_MAX_K1_KERNEL(min_max_fp16_k1_avx2, f16x16_rvar, f16x16_rvar, half, 16);
MIN_MAX_KN_KERNEL(min_max_fp16_kn_avx2, f16x16_rvar, f16x16_rvar, half, 16);
MIN_MAX_K1_KERNEL(min_max_uint8_k1_avx2, u8x32, u8x32, uint8_t, 32);
MIN_MAX_KN_KERNEL(min_max_uint8_kn_avx2, u8x32, u8x32, uint8_t, 32);
MIN_MAX_K1_KERNEL(min_max_int8_k1_avx2, s8x32, s8x32, int8_t, 32);
MIN_MAX_KN_KERNEL(min_max_int8_kn_avx2, s8x32, s8x32, int8_t, 32);

MIN_MAX_K1_KERNEL(min_bf16_k1_avx2, bf16x16_rvar, dummy_t, bfloat16, 16);
MIN_MAX_KN_KERNEL(min_bf16_kn_avx2, bf16x16_rvar, dummy_t, bfloat16, 16);
MIN_MAX_K1_KERNEL(min_fp16_k1_avx2, f16x16_rvar, dummy_t, half, 16);
MIN_MAX_KN_KERNEL(min_fp16_kn_avx2, f16x16_rvar, dummy_t, half, 16);
MIN_MAX_K1_KERNEL(min_uint8_k1_avx2, u8x32, dummy_t, uint8_t, 32);
MIN_MAX_KN_KERNEL(min_uint8_kn_avx2, u8x32, dummy_t, uint8_t, 32);
MIN_MAX_K1_KERNEL(min_int8_k1_avx2, s8x32, dummy_t, int8_t, 32);
MIN_MAX_KN_KERNEL(min_int8_kn_avx2, s8x32, dummy_t, int8_t, 32);

MIN_MAX_K1_KERNEL(max_bf16_k1_avx2, dummy_t, bf16x16_rvar, bfloat16, 16);
MIN_MAX_KN_KERNEL(max_bf16_kn_avx2, dummy_t, bf16x16_rvar, bfloat16, 16);
MIN_MAX_K1_KERNEL(max_fp16_k1_avx2, dummy_t, f16x16_rvar, half, 16);
MIN_MAX_KN_KERNEL(max_fp16_kn_avx2, dummy_t, f16x16_rvar, half, 16);
MIN_MAX_K1_KERNEL(max_uint8_k1_avx2, dummy_t, u8x32, uint8_t, 32);
MIN_MAX_KN_KERNEL(max_uint8_kn_avx2, dummy_t, u8x32, uint8_t, 32);
MIN_MAX_K1_KERNEL(max_int8_k1_avx2, dummy_t, s8x32, int8_t, 32);
MIN_MAX_KN_KERNEL(max_int8_kn_avx2, dummy_t, s8x32, int8_t, 32);

SUM_K1_KERNEL(sum_int8_int32_k1_avx2, int8_t, int32_t, 8, 4, identity);
SUM_KN_KERNEL(sum_int8_int32_kn_avx2, int8_t, int32_t, 32, identity);
SUM_K1_KERNEL(sum_uint8_int32_k1_avx2, uint8_t, int32_t, 8, 4, identity);
SUM_KN_KERNEL(sum_uint8_int32_kn_avx2, uint8_t, int32_t, 32, identity);
SUM_K1_KERNEL(sum_int32_k1_avx2, int32_t, int32_t, 8, 1, identity);
SUM_KN_KERNEL(sum_int32_kn_avx2, int32_t, int32_t, 8, identity);
SUM_K1_KERNEL(sum_bf16_fp32_k1_avx2, bfloat16, float, consistent_tile_k_fp32, 2,
              identity);
SUM_KN_KERNEL(sum_bf16_fp32_kn_avx2, bfloat16, float, 16, identity);

SUM_K1_KERNEL(sum_squared_int8_int32_k1_avx2, int8_t, int32_t, 8, 2, square);
SUM_KN_KERNEL(sum_squared_int8_int32_kn_avx2, int8_t, int32_t, 32, square);
SUM_K1_KERNEL(sum_squared_uint8_int32_k1_avx2, uint8_t, int32_t, 8, 2, square);
SUM_KN_KERNEL(sum_squared_uint8_int32_kn_avx2, uint8_t, int32_t, 32, square);
SUM_K1_KERNEL(sum_squared_bf16_fp32_k1_avx2, bfloat16, float,
              consistent_tile_k_fp32, 2, square);
SUM_KN_KERNEL(sum_squared_bf16_fp32_kn_avx2, bfloat16, float, 16, square);

}  // namespace ynn
