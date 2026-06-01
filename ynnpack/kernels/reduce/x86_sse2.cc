// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/simd/x86_vec128.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum.h"

namespace ynn {

namespace simd {

// Use psadbw to compute the absolute difference of a and 0, summing 8 of them
// and producing an int64 in their place. We reinterpret the result to be 4
// int32s, which is only correct because we will do a horizontal total reduction
// later.
static s32x4 reduce_add(
    s32x4 a, u8x16 b, identity /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  s32x4 b_s32(_mm_sad_epu8(b.v, _mm_set1_epi8(0)));
  return a += b_s32;
}

template <typename MapFn>
static f32x4 reduce_add(
    f32x4 a, bf16x8 b, MapFn map_fn,
    std::integral_constant<size_t, 2> /*horizontal_factor*/ = {}) {
  __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0xFFFF0000));
  f32x4 evens(_mm_castsi128_ps(_mm_slli_epi32(b.v, 16)));
  f32x4 odds(_mm_and_ps(_mm_castsi128_ps(b.v), mask));
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
  f32x4 a0 =
      reduce_add(extract<0>(a, f32x4::N), extract<0>(b, bf16x8::N), map_fn);
  f32x4 a1 =
      reduce_add(extract<1>(a, f32x4::N), extract<1>(b, bf16x8::N), map_fn);
  f32x4 a2 =
      reduce_add(extract<2>(a, f32x4::N), extract<2>(b, bf16x8::N), map_fn);
  f32x4 a3 =
      reduce_add(extract<3>(a, f32x4::N), extract<3>(b, bf16x8::N), map_fn);
  return {{a0, a1}, {a2, a3}};
}

}  // namespace simd

namespace {

using simd::bf16x8;
using simd::f16x8;
using simd::f32x4;
using simd::f32x8;
using simd::s16x8;
using simd::s32x16;
using simd::s32x4;
using simd::s8x16;
using simd::u8x16;

using f16x8_rvar = float16_wrapper<f16x8, s16x8>;
using bf16x8_rvar = float16_wrapper<bf16x8, s16x8>;

}  // namespace

MIN_MAX_K1_KERNEL(min_max_k1_fp32_sse2, f32x4, f32x4, float, 4);
MIN_MAX_KN_KERNEL(min_max_kn_fp32_sse2, f32x4, f32x4, float, 4);
MIN_MAX_K1_KERNEL(min_max_k1_bf16_sse2, bf16x8_rvar, bf16x8_rvar, bfloat16, 8);
MIN_MAX_KN_KERNEL(min_max_kn_bf16_sse2, bf16x8_rvar, bf16x8_rvar, bfloat16, 8);
MIN_MAX_K1_KERNEL(min_max_k1_fp16_sse2, f16x8_rvar, f16x8_rvar, half, 8);
MIN_MAX_KN_KERNEL(min_max_kn_fp16_sse2, f16x8_rvar, f16x8_rvar, half, 8);
MIN_MAX_K1_KERNEL(min_max_k1_uint8_sse2, u8x16, u8x16, uint8_t, 16);
MIN_MAX_KN_KERNEL(min_max_kn_uint8_sse2, u8x16, u8x16, uint8_t, 16);

MIN_MAX_K1_KERNEL(min_k1_fp32_sse2, f32x4, dummy_t, float, 4);
MIN_MAX_KN_KERNEL(min_kn_fp32_sse2, f32x4, dummy_t, float, 4);
MIN_MAX_K1_KERNEL(min_k1_bf16_sse2, bf16x8_rvar, dummy_t, bfloat16, 8);
MIN_MAX_KN_KERNEL(min_kn_bf16_sse2, bf16x8_rvar, dummy_t, bfloat16, 8);
MIN_MAX_K1_KERNEL(min_k1_fp16_sse2, f16x8_rvar, dummy_t, half, 8);
MIN_MAX_KN_KERNEL(min_kn_fp16_sse2, f16x8_rvar, dummy_t, half, 8);
MIN_MAX_K1_KERNEL(min_k1_uint8_sse2, u8x16, dummy_t, uint8_t, 16);
MIN_MAX_KN_KERNEL(min_kn_uint8_sse2, u8x16, dummy_t, uint8_t, 16);

MIN_MAX_K1_KERNEL(max_k1_fp32_sse2, dummy_t, f32x4, float, 4);
MIN_MAX_KN_KERNEL(max_kn_fp32_sse2, dummy_t, f32x4, float, 4);
MIN_MAX_K1_KERNEL(max_k1_bf16_sse2, dummy_t, bf16x8_rvar, bfloat16, 8);
MIN_MAX_KN_KERNEL(max_kn_bf16_sse2, dummy_t, bf16x8_rvar, bfloat16, 8);
MIN_MAX_K1_KERNEL(max_k1_fp16_sse2, dummy_t, f16x8_rvar, half, 8);
MIN_MAX_KN_KERNEL(max_kn_fp16_sse2, dummy_t, f16x8_rvar, half, 8);
MIN_MAX_K1_KERNEL(max_k1_uint8_sse2, dummy_t, u8x16, uint8_t, 16);
MIN_MAX_KN_KERNEL(max_kn_uint8_sse2, dummy_t, u8x16, uint8_t, 16);

SUM_FLOAT_K1_KERNEL(sum_k1_bf16_fp32_sse2, bfloat16, float, 0, 2, identity);
SUM_FLOAT_KN_KERNEL(sum_kn_bf16_fp32_sse2, bfloat16, float, 8, identity);
SUM_FLOAT_K1_KERNEL(sum_k1_fp32_sse2, float, float, 0, 1, identity);
SUM_FLOAT_KN_KERNEL(sum_kn_fp32_sse2, float, float, 4, identity);
SUM_KN_KERNEL(sum_kn_int8_int32_sse2, int8_t, int32_t, 16, identity);
SUM_K1_KERNEL(sum_k1_uint8_int32_sse2, uint8_t, int32_t, 4, 4, identity);
SUM_KN_KERNEL(sum_kn_uint8_int32_sse2, uint8_t, int32_t, 16, identity);
SUM_K1_KERNEL(sum_k1_int32_sse2, int32_t, int32_t, 4, 1, identity);
SUM_KN_KERNEL(sum_kn_int32_sse2, int32_t, int32_t, 4, identity);

SUM_FLOAT_K1_KERNEL(sum_squared_k1_bf16_fp32_sse2, bfloat16, float, 0, 2,
                    square);
SUM_FLOAT_KN_KERNEL(sum_squared_kn_bf16_fp32_sse2, bfloat16, float, 8, square);
SUM_FLOAT_K1_KERNEL(sum_squared_k1_fp32_sse2, float, float, 0, 1, square);
SUM_FLOAT_KN_KERNEL(sum_squared_kn_fp32_sse2, float, float, 4, square);

}  // namespace ynn
