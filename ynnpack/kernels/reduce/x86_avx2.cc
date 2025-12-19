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
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max_accumulator.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace simd {

static s32x16 reduce_add(
    s32x16 a, u8x16 b, Square /*map_fn*/,
    std::integral_constant<size_t, 1>/*horizontal_factor*/) {
  __m256i b_lo = _mm256_cvtepu8_epi32(b.v);
  __m256i b_hi = _mm256_cvtepu8_epi32(_mm_bsrli_si128(b.v, 8));

  // madd_epi16 works due to extra zeros from uint8 -> int32 conversion.
  a[0] += s32x8{_mm256_madd_epi16(b_lo, b_lo)};
  a[1] += s32x8{_mm256_madd_epi16(b_hi, b_hi)};
  return a;
}

static s32x16 reduce_add(s32x16 a, s8x16 b, Square map_fn,
                          std::integral_constant<size_t, 1> horizontal_factor) {
  // We're squaring, we can take the absolute value and use the unsigned reduce.
  return reduce_add(a, abs(b), map_fn, horizontal_factor);
}

static s32x8 reduce_add(
    s32x8 a, s8x32 b, Identity /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  __m256i b2x = _mm256_maddubs_epi16(_mm256_set1_epi8(1), b.v);
  s32x8 b_s32(_mm256_madd_epi16(_mm256_set1_epi16(1), b2x));
  return a += b_s32;
}

static s32x8 reduce_add(
    s32x8 a, u8x32 b, Identity /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  __m256i b2x = _mm256_maddubs_epi16(b.v, _mm256_set1_epi8(1));
  s32x8 b_s32(_mm256_madd_epi16(_mm256_set1_epi16(1), b2x));
  return a += b_s32;
}

static s32x8 reduce_add(
    s32x8 a, s8x16 b, Square /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  __m256i b_16 = _mm256_cvtepi8_epi16(b.v);
  return a += s32x8(_mm256_madd_epi16(b_16, b_16));
}

static s32x8 reduce_add(
    s32x8 a, u8x16 b, Square /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  __m256i b_16 = _mm256_cvtepu8_epi16(b.v);
  return a += s32x8(_mm256_madd_epi16(b_16, b_16));
}

template <typename MapFn>
static f32x8 reduce_add(
    f32x8 a, bf16x16 b, MapFn map_fn,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0xFFFF0000));
  f32x8 evens(_mm256_castsi256_ps(_mm256_slli_epi32(b.v, 16)));
  f32x8 odds(_mm256_and_ps(_mm256_castsi256_ps(b.v), mask));

  a += map_fn(odds);
  a += map_fn(evens);
  return a;
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

MIN_MAX_KERNEL(min_max_bf16_4x16_avx2, bf16x16_rvar, bf16x16_rvar, bfloat16,
               16);
MIN_MAX_KERNEL(min_max_fp16_4x16_avx2, f16x16_rvar, f16x16_rvar, half, 16);
MIN_MAX_KERNEL(min_max_uint8_4x32_avx2, u8x32, u8x32, uint8_t, 32);
MIN_MAX_KERNEL(min_max_int8_4x32_avx2, s8x32, s8x32, int8_t, 32);

MIN_MAX_KERNEL(min_bf16_4x16_avx2, bf16x16_rvar, dummy_t, bfloat16, 16);
MIN_MAX_KERNEL(min_fp16_4x16_avx2, f16x16_rvar, dummy_t, half, 16);
MIN_MAX_KERNEL(min_uint8_4x32_avx2, u8x32, dummy_t, uint8_t, 32);
MIN_MAX_KERNEL(min_int8_4x32_avx2, s8x32, dummy_t, int8_t, 32);

MIN_MAX_KERNEL(max_bf16_4x16_avx2, dummy_t, bf16x16_rvar, bfloat16, 16);
MIN_MAX_KERNEL(max_fp16_4x16_avx2, dummy_t, f16x16_rvar, half, 16);
MIN_MAX_KERNEL(max_uint8_4x32_avx2, dummy_t, u8x32, uint8_t, 32);
MIN_MAX_KERNEL(max_int8_4x32_avx2, dummy_t, s8x32, int8_t, 32);

void sum_int8_int32_avx2(size_t n, size_t k3, size_t k2, size_t k1,
                         size_t a_stride_n, size_t a_stride_k3,
                         size_t a_stride_k2, const void* a, size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(int8_t)) {
    stream_reduce<sum_accumulator_k1_1<s32x32>, int8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const int8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x8, 32>, int8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const int8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

void sum_uint8_int32_avx2(size_t n, size_t k3, size_t k2, size_t k1,
                          size_t a_stride_n, size_t a_stride_k3,
                          size_t a_stride_k2, const void* a, size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(uint8_t)) {
    stream_reduce<sum_accumulator_k1_1<s32x32>, uint8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x8, 32>, uint8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

void sum_squared_int8_int32_avx2(size_t n, size_t k3, size_t k2, size_t k1,
                                 size_t a_stride_n, size_t a_stride_k3,
                                 size_t a_stride_k2, const void* a, size_t,
                                 void* c) {
  if (k1 == 1 && a_stride_n == sizeof(int8_t)) {
    stream_reduce<sum_accumulator_k1_1<s32x16, Square>, int8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const int8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x8, 16, Square>, int8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const int8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

void sum_squared_uint8_int32_avx2(size_t n, size_t k3, size_t k2, size_t k1,
                                  size_t a_stride_n, size_t a_stride_k3,
                                  size_t a_stride_k2, const void* a, size_t,
                                  void* c) {
  if (k1 == 1 && a_stride_n == sizeof(uint8_t)) {
    stream_reduce<sum_accumulator_k1_1<s32x16, Square>, uint8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x8, 16, Square>, uint8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

void sum_bf16_fp32_avx2(size_t n, size_t k3, size_t k2, size_t k1,
                        size_t a_stride_n, size_t a_stride_k3,
                        size_t a_stride_k2, const void* a, size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(bfloat16)) {
    stream_reduce<sum_accumulator_k1_1<f32x8>, bfloat16, float>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<f32x8, 16>, bfloat16, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

void sum_squared_bf16_fp32_avx2(size_t n, size_t k3, size_t k2, size_t k1,
                                size_t a_stride_n, size_t a_stride_k3,
                                size_t a_stride_k2, const void* a, size_t,
                                void* c) {
  if (k1 == 1 && a_stride_n == sizeof(bfloat16)) {
    stream_reduce<sum_accumulator_k1_1<f32x8, Square>, bfloat16, float>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<f32x8, 16, Square>, bfloat16, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

}  // namespace ynn
