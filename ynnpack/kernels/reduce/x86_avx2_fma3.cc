// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_avx2.h"

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <cstring>
#include <type_traits>

#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/simd/multi_vec.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace simd {

using f32x8x16 = simd::multi_vec<f32x8, 16>;
using bf16x16x8 = multi_vec<bf16x16, 8>;

static f32x8x16 reduce_add(
    f32x8x16 a, bf16x16x8 b, Identity /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  YNN_UNROLL
  for (int i = 0; i < 8; ++i) {
    __m256i lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b.v[i].v));
    __m256i hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(b.v[i].v, 1));

    a.v[2 * i + 0] += f32x8{_mm256_castsi256_ps(_mm256_slli_epi32(lo, 16))};
    a.v[2 * i + 1] += f32x8{_mm256_castsi256_ps(_mm256_slli_epi32(hi, 16))};
  }

  return a;
}

static f32x8 reduce_add(
    f32x8 a, bf16x16 b, Identity /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0xFFFF0000));
  __m256 evens = _mm256_castsi256_ps(_mm256_slli_epi32(b.v, 16));
  __m256 odds = _mm256_and_ps(_mm256_castsi256_ps(b.v), mask);

  a += f32x8{odds};
  a += f32x8{evens};
  return a;
}

static f32x8x16 reduce_add(
    f32x8x16 a, bf16x16x8 b, Square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  YNN_UNROLL
  for (int i = 0; i < 8; ++i) {
    __m256i lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b.v[i].v));
    __m256i hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(b.v[i].v, 1));
    __m256 lo_f32 = _mm256_castsi256_ps(_mm256_slli_epi32(lo, 16));
    __m256 hi_f32 = _mm256_castsi256_ps(_mm256_slli_epi32(hi, 16));

    a.v[2 * i + 0].v = _mm256_fmadd_ps(lo_f32, lo_f32, a.v[2 * i + 0].v);
    a.v[2 * i + 1].v = _mm256_fmadd_ps(hi_f32, hi_f32, a.v[2 * i + 1].v);
  }

  return a;
}

static f32x8 reduce_add(
    f32x8 a, bf16x16 b, Square /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0xFFFF0000));
  __m256 evens = _mm256_castsi256_ps(_mm256_slli_epi32(b.v, 16));
  __m256 odds = _mm256_and_ps(_mm256_castsi256_ps(b.v), mask);
  a.v = _mm256_fmadd_ps(odds, odds, a.v);
  a.v = _mm256_fmadd_ps(evens, evens, a.v);
  return a;
}

}  // namespace simd

using simd::f32x8;
using simd::f32x8x16;
using simd::bf16x16;
using simd::bf16x16x8;
using simd::s16x16;

using bf16x16_rvar = float16_wrapper<bf16x16, s16x16>;

void sum_bf16_fp32_avx2_fma3(size_t n, size_t k3, size_t k2, size_t k1,
                             size_t a_stride_n, size_t a_stride_k3,
                             size_t a_stride_k2, const void* a, size_t,
                             void* c) {
  if (k1 == 1 && a_stride_n == sizeof(bfloat16)) {
    tiled_reduce<sum_accumulator_k1_1<bf16x16x8, f32x8x16>, bfloat16, float>(
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

void sum_squared_bf16_fp32_avx2_fma3(size_t n, size_t k3, size_t k2, size_t k1,
                                     size_t a_stride_n, size_t a_stride_k3,
                                     size_t a_stride_k2, const void* a, size_t,
                                     void* c) {
  if (k1 == 1 && a_stride_n == sizeof(bfloat16)) {
    tiled_reduce<sum_accumulator_k1_1<bf16x16x8, f32x8x16, Square>, bfloat16,
      float>(
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
