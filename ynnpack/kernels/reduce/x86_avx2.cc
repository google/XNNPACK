// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_avx2.h"

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/multi_vec.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max_accumulator.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace simd {

using s32x8x4 = multi_vec<s32x8, 4>;

static s32x8x4& operator+=(s32x8x4& a, s8x32 b) {
  s8x16 b_lo = extract<0>(b, s8x16{});
  s8x16 b_hi = extract<1>(b, s8x16{});
  s32x8 b_0(_mm256_cvtepi8_epi32(b_lo.v));
  s32x8 b_1(_mm256_cvtepi8_epi32(_mm_srli_si128(b_lo.v, 8)));
  s32x8 b_2(_mm256_cvtepi8_epi32(b_hi.v));
  s32x8 b_3(_mm256_cvtepi8_epi32(_mm_srli_si128(b_hi.v, 8)));

  a.v[0] += b_0;
  a.v[1] += b_1;
  a.v[2] += b_2;
  a.v[3] += b_3;
  return a;
}

static s32x8x4& operator+=(s32x8x4& a, u8x32 b) {
  u8x16 b_lo = extract<0>(b, u8x16{});
  u8x16 b_hi = extract<1>(b, u8x16{});
  s32x8 b_0(_mm256_cvtepu8_epi32(b_lo.v));
  s32x8 b_1(_mm256_cvtepu8_epi32(_mm_srli_si128(b_lo.v, 8)));
  s32x8 b_2(_mm256_cvtepu8_epi32(b_hi.v));
  s32x8 b_3(_mm256_cvtepu8_epi32(_mm_srli_si128(b_hi.v, 8)));

  a.v[0] += b_0;
  a.v[1] += b_1;
  a.v[2] += b_2;
  a.v[3] += b_3;
  return a;
}

static s32x8& reduce_add(
    s32x8& a, s8x32 b,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  __m256i b2x = _mm256_maddubs_epi16(_mm256_set1_epi8(1), b.v);
  s32x8 b_s32(_mm256_madd_epi16(_mm256_set1_epi16(1), b2x));
  return a += b_s32;
}

static s32x8& reduce_add(
    s32x8& a, u8x32 b,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  __m256i b2x = _mm256_maddubs_epi16(b.v, _mm256_set1_epi8(1));
  s32x8 b_s32(_mm256_madd_epi16(_mm256_set1_epi16(1), b2x));
  return a += b_s32;
}

}  // namespace simd

using simd::s32x8;
using simd::s32x8x4;
using simd::f32x8;
using simd::bf16x16;
using simd::f16x16;
using simd::s16x16;
using simd::s8x32;
using simd::u8x32;

using f16x16_rvar = float16_wrapper<f16x16, s16x16>;
using bf16x16_rvar = float16_wrapper<bf16x16, s16x16>;

MIN_MAX_KERNEL(min_max_fp32_4x8_avx2, f32x8, f32x8, float, 8);
MIN_MAX_KERNEL(min_max_bf16_4x16_avx2, bf16x16_rvar, bf16x16_rvar, bfloat16,
               16);
MIN_MAX_KERNEL(min_max_fp16_4x16_avx2, f16x16_rvar, f16x16_rvar, half, 16);
MIN_MAX_KERNEL(min_max_uint8_4x32_avx2, u8x32, u8x32, uint8_t, 32);
MIN_MAX_KERNEL(min_max_int8_4x32_avx2, s8x32, s8x32, int8_t, 32);

MIN_MAX_KERNEL(min_fp32_4x8_avx2, f32x8, dummy_t, float, 8);
MIN_MAX_KERNEL(min_bf16_4x16_avx2, bf16x16_rvar, dummy_t, bfloat16, 16);
MIN_MAX_KERNEL(min_fp16_4x16_avx2, f16x16_rvar, dummy_t, half, 16);
MIN_MAX_KERNEL(min_uint8_4x32_avx2, u8x32, dummy_t, uint8_t, 32);
MIN_MAX_KERNEL(min_int8_4x32_avx2, s8x32, dummy_t, int8_t, 32);

MIN_MAX_KERNEL(max_fp32_4x8_avx2, dummy_t, f32x8, float, 8);
MIN_MAX_KERNEL(max_bf16_4x16_avx2, dummy_t, bf16x16_rvar, bfloat16, 16);
MIN_MAX_KERNEL(max_fp16_4x16_avx2, dummy_t, f16x16_rvar, half, 16);
MIN_MAX_KERNEL(max_uint8_4x32_avx2, dummy_t, u8x32, uint8_t, 32);
MIN_MAX_KERNEL(max_int8_4x32_avx2, dummy_t, s8x32, int8_t, 32);

void sum_int8_int32_avx2(size_t n, size_t k3, size_t k2, size_t k1,
                         size_t a_stride_n, size_t a_stride_k3,
                         size_t a_stride_k2, const void* a, size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(int8_t)) {
    tiled_reduce<sum_accumulator_k1_1<s8x32, s32x8x4>, int8_t, int32_t>(
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
    tiled_reduce<sum_accumulator_k1_1<u8x32, s32x8x4>, uint8_t, int32_t>(
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

using f32x8x8 = simd::multi_vec<f32x8, 8>;

void sum_fp32_avx2(size_t n, size_t k3, size_t k2, size_t k1,
                   size_t a_stride_n, size_t a_stride_k3, size_t a_stride_k2,
                   const void* a, size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(float)) {
    tiled_reduce<sum_accumulator_k1_1<f32x8x8, f32x8x8>, float, float>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const float*>(a),
        /*C_stride_m=*/0, reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<f32x8, 8>, float, float>(
      n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
      reinterpret_cast<const float*>(a), /*C_stride_m=*/0,
      reinterpret_cast<float*>(c));
  }
}

}  // namespace ynn
