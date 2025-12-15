// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <cstddef>
#include <type_traits>

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/simd/multi_vec.h"
#include "ynnpack/base/simd/x86_avx2.h"
#include "ynnpack/base/simd/x86_fma3.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace simd {

using f32x16x8 = simd::multi_vec<f32x16, 8>;
using bf16x16x8 = multi_vec<bf16x16, 8>;

static f32x16x8 reduce_add(
    f32x16x8 a, bf16x16x8 b, Square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  auto b_f32 = convert(b, float{});
  a = fma(b_f32, b_f32, a);
  return a;
}

static f32x8 reduce_add(
    f32x8 a, bf16x16 b, Identity /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0xFFFF0000));
  f32x8 evens{_mm256_castsi256_ps(_mm256_slli_epi32(b.v, 16))};
  f32x8 odds{_mm256_and_ps(_mm256_castsi256_ps(b.v), mask)};
  a += odds;
  a += evens;
  return a;
}

static f32x8 reduce_add(
    f32x8 a, bf16x16 b, Square /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0xFFFF0000));
  f32x8 evens{_mm256_castsi256_ps(_mm256_slli_epi32(b.v, 16))};
  f32x8 odds{_mm256_and_ps(_mm256_castsi256_ps(b.v), mask)};
  a = fma(odds, odds, a);
  a = fma(evens, evens, a);
  return a;
}

}  // namespace simd

using simd::bf16x16;
using simd::bf16x16x8;
using simd::f32x16x8;
using simd::f32x8;
using simd::s16x16;

void sum_bf16_fp32_avx2_fma3(size_t n, size_t k3, size_t k2, size_t k1,
                             size_t a_stride_n, size_t a_stride_k3,
                             size_t a_stride_k2, const void* a, size_t,
                             void* c) {
  if (k1 == 1 && a_stride_n == sizeof(bfloat16)) {
    tiled_reduce<sum_accumulator_k1_1<bf16x16x8, f32x16x8>, bfloat16, float>(
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
    tiled_reduce<sum_accumulator_k1_1<bf16x16x8, f32x16x8, Square>, bfloat16,
                 float>(n, k3, k2, a_stride_k3, a_stride_k2,
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
