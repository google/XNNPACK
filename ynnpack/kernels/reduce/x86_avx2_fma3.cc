// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <cstddef>
#include <type_traits>

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/simd/x86_avx2.h"
#include "ynnpack/base/simd/x86_fma3.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace simd {

static f32x8 reduce_add(
    f32x8 a, bf16x16 b, Square /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/ = {}) {
  __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0xFFFF0000));
  f32x8 evens{_mm256_castsi256_ps(_mm256_slli_epi32(b.v, 16))};
  f32x8 odds{_mm256_and_ps(_mm256_castsi256_ps(b.v), mask)};
  a = fma(odds, odds, a);
  a = fma(evens, evens, a);
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
using simd::f32x8;

void sum_squared_bf16_fp32_avx2_fma3(size_t n, size_t k3, size_t k2, size_t k1,
                                     size_t a_stride_n, size_t a_stride_k3,
                                     size_t a_stride_k2, const void* a, size_t,
                                     void* c) {
  if (k1 == 1 && a_stride_n == sizeof(bfloat16)) {
    stream_reduce<sum_accumulator_k1_1<f32x8, Square>, bfloat16, float>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_fp32<2, Square>, bfloat16, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

}  // namespace ynn
