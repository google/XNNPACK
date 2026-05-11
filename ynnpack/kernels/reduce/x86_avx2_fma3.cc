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
#include "ynnpack/kernels/reduce/sum.h"

namespace ynn {

namespace simd {

static f32x8 reduce_add(
    f32x8 a, bf16x16 b, square /*map_fn*/,
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

SUM_FLOAT_K1_KERNEL(sum_k1_bf16_fp32_avx2_fma3, bfloat16, float, 0, 2, square);
SUM_FLOAT_KN_KERNEL(sum_kn_bf16_fp32_avx2_fma3, bfloat16, float, 16, square);

}  // namespace ynn
