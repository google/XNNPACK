// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/simd/x86_vec512.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum.h"

namespace ynn {

namespace simd {

static s32x16 reduce_add(
    s32x16 a, s8x64 b, identity /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  __m512i b2x = _mm512_maddubs_epi16(_mm512_set1_epi8(1), b.v);
  s32x16 b_s32(_mm512_madd_epi16(_mm512_set1_epi16(1), b2x));
  return a += b_s32;
}

static s32x16 reduce_add(
    s32x16 a, u8x64 b, identity /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  __m512i b2x = _mm512_maddubs_epi16(b.v, _mm512_set1_epi8(1));
  s32x16 b_s32(_mm512_madd_epi16(_mm512_set1_epi16(1), b2x));
  return a += b_s32;
}

static s32x64 reduce_add(
    s32x64 a, u8x64 b, square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  s32x64 b_s32 = cast(b, int32_t{});
  s32x16 b0 = extract<0>(b_s32, s32x16::N);
  s32x16 b1 = extract<1>(b_s32, s32x16::N);
  s32x16 b2 = extract<2>(b_s32, s32x16::N);
  s32x16 b3 = extract<3>(b_s32, s32x16::N);

  // madd_epi16 works due to extra zeros from uint8 -> int32 conversion.
  return a + concat(s32x16{_mm512_madd_epi16(b0.v, b0.v)},
                    s32x16{_mm512_madd_epi16(b1.v, b1.v)},
                    s32x16{_mm512_madd_epi16(b2.v, b2.v)},
                    s32x16{_mm512_madd_epi16(b3.v, b3.v)});
}

static s32x64 reduce_add(s32x64 a, s8x64 b, square map_fn,
                         std::integral_constant<size_t, 1> horizontal_factor) {
  // We're squaring, we can take the absolute value and use the unsigned reduce.
  return reduce_add(a, abs(b), map_fn, horizontal_factor);
}

static s32x16 reduce_add(
    s32x16 a, s8x64 b, square /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  __m512i b_even = _mm512_and_si512(b.v, _mm512_set1_epi16(0x00FF));
  // Negative values won't be sign extended by the mask above. Sign extension is
  // expensive, and since we're squaring, we can just take the absolute value
  // instead.
  b_even = _mm512_abs_epi8(b_even);
  __m512i b_odd = _mm512_srai_epi16(b.v, 8);
  a += s32x16(_mm512_madd_epi16(b_even, b_even));
  a += s32x16(_mm512_madd_epi16(b_odd, b_odd));
  return a;
}

static s32x16 reduce_add(
    s32x16 a, u8x64 b, square /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  __m512i b_even = _mm512_and_si512(b.v, _mm512_set1_epi16(0x00FF));
  __m512i b_odd = _mm512_srli_epi16(b.v, 8);
  a += s32x16(_mm512_madd_epi16(b_even, b_even));
  a += s32x16(_mm512_madd_epi16(b_odd, b_odd));
  return a;
}

static f32x16 reduce_add(
    f32x16 a, bf16x32 b, identity /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  __m512i mask = _mm512_set1_epi32(0xFFFF0000);
  f32x16 evens(_mm512_castsi512_ps(_mm512_slli_epi32(b.v, 16)));
  f32x16 odds(_mm512_castsi512_ps(_mm512_and_epi32(b.v, mask)));

  a += odds;
  a += evens;
  return a;
}

static f32x32 reduce_add(
    f32x32 a, bf16x32 b, square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  f32x32 b_f32 = cast(b, float{});
  return fma(b_f32, b_f32, a);
}

static f32x16 reduce_add(
    f32x16 a, bf16x32 b, square /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  __m512i mask = _mm512_set1_epi32(0xFFFF0000);
  f32x16 evens{_mm512_castsi512_ps(_mm512_slli_epi32(b.v, 16))};
  f32x16 odds{_mm512_castsi512_ps(_mm512_and_epi32(b.v, mask))};

  a = fma(odds, odds, a);
  a = fma(evens, evens, a);
  return a;
}

static f32x32 reduce_add(
    f32x32 a, f16x32 b, square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  f32x32 b_f32 = cast(b, float{});
  return fma(b_f32, b_f32, a);
}

static f32x16 reduce_add(
    f32x16 a, f16x16 b, square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  f32x16 b_f32 = cast(b, float{});
  return fma(b_f32, b_f32, a);
}

YNN_ALWAYS_INLINE s8x64 sign_complement(s8x64 x) {
  __m512i zero = _mm512_setzero_si512();
  __mmask64 sign_mask = _mm512_cmpgt_epi8_mask(zero, x.v);
  __m512i sign = _mm512_movm_epi8(sign_mask);
  __m512i mask = _mm512_set1_epi8(0x7F);
  __m512i abs_val = _mm512_and_si512(x.v, mask);
  return s8x64(_mm512_xor_si512(abs_val, sign));
}

}  // namespace simd

using simd::bf16x32;
using simd::f16x32;
using simd::f32x16;
using simd::f32x32;
using simd::f64x8;
using simd::s16x32;
using simd::s32x16;
using simd::s32x32;
using simd::s32x64;
using simd::s32x8;
using simd::s8x16;
using simd::s8x32;
using simd::s8x64;
using simd::u8x16;
using simd::u8x32;
using simd::u8x64;

using xf16x32_rvar = sign_magnitude<s16x32>;
using xf8x64_rvar = sign_magnitude<s8x64>;

MIN_MAX_K1_KERNEL(min_max_k1_fp32_avx512, f32x16, f32x16, float, 16);
MIN_MAX_KN_KERNEL(min_max_kn_fp32_avx512, f32x16, f32x16, float, 16);
MIN_MAX_K1_KERNEL(min_max_k1_xf16_avx512, xf16x32_rvar, xf16x32_rvar, int16_t,
                  32);
MIN_MAX_KN_KERNEL(min_max_kn_xf16_avx512, xf16x32_rvar, xf16x32_rvar, int16_t,
                  32);
MIN_MAX_K1_KERNEL(min_max_k1_xf8_avx512, xf8x64_rvar, xf8x64_rvar, int8_t, 64);
MIN_MAX_KN_KERNEL(min_max_kn_xf8_avx512, xf8x64_rvar, xf8x64_rvar, int8_t, 64);
MIN_MAX_K1_KERNEL(min_max_k1_uint8_avx512, u8x64, u8x64, uint8_t, 64);
MIN_MAX_KN_KERNEL(min_max_kn_uint8_avx512, u8x64, u8x64, uint8_t, 64);
MIN_MAX_K1_KERNEL(min_max_k1_int8_avx512, s8x64, s8x64, int8_t, 64);
MIN_MAX_KN_KERNEL(min_max_kn_int8_avx512, s8x64, s8x64, int8_t, 64);

MIN_MAX_K1_KERNEL(min_k1_fp32_avx512, f32x16, dummy_t, float, 16);
MIN_MAX_KN_KERNEL(min_kn_fp32_avx512, f32x16, dummy_t, float, 16);
MIN_MAX_K1_KERNEL(min_k1_xf16_avx512, xf16x32_rvar, dummy_t, int16_t, 32);
MIN_MAX_KN_KERNEL(min_kn_xf16_avx512, xf16x32_rvar, dummy_t, int16_t, 32);
MIN_MAX_K1_KERNEL(min_k1_xf8_avx512, xf8x64_rvar, dummy_t, int8_t, 64);
MIN_MAX_KN_KERNEL(min_kn_xf8_avx512, xf8x64_rvar, dummy_t, int8_t, 64);
MIN_MAX_K1_KERNEL(min_k1_uint8_avx512, u8x64, dummy_t, uint8_t, 64);
MIN_MAX_KN_KERNEL(min_kn_uint8_avx512, u8x64, dummy_t, uint8_t, 64);
MIN_MAX_K1_KERNEL(min_k1_int8_avx512, s8x64, dummy_t, int8_t, 64);
MIN_MAX_KN_KERNEL(min_kn_int8_avx512, s8x64, dummy_t, int8_t, 64);

MIN_MAX_K1_KERNEL(max_k1_fp32_avx512, dummy_t, f32x16, float, 16);
MIN_MAX_KN_KERNEL(max_kn_fp32_avx512, dummy_t, f32x16, float, 16);
MIN_MAX_K1_KERNEL(max_k1_xf16_avx512, dummy_t, xf16x32_rvar, int16_t, 32);
MIN_MAX_KN_KERNEL(max_kn_xf16_avx512, dummy_t, xf16x32_rvar, int16_t, 32);
MIN_MAX_K1_KERNEL(max_k1_xf8_avx512, dummy_t, xf8x64_rvar, int8_t, 64);
MIN_MAX_KN_KERNEL(max_kn_xf8_avx512, dummy_t, xf8x64_rvar, int8_t, 64);
MIN_MAX_K1_KERNEL(max_k1_uint8_avx512, dummy_t, u8x64, uint8_t, 64);
MIN_MAX_KN_KERNEL(max_kn_uint8_avx512, dummy_t, u8x64, uint8_t, 64);
MIN_MAX_K1_KERNEL(max_k1_int8_avx512, dummy_t, s8x64, int8_t, 64);
MIN_MAX_KN_KERNEL(max_kn_int8_avx512, dummy_t, s8x64, int8_t, 64);

MIN_MAX_K1_KERNEL(min_max_k1_fp64_avx512, f64x8, f64x8, double, 8);
MIN_MAX_KN_KERNEL(min_max_kn_fp64_avx512, f64x8, f64x8, double, 8);
MIN_MAX_K1_KERNEL(min_k1_fp64_avx512, f64x8, dummy_t, double, 8);
MIN_MAX_KN_KERNEL(min_kn_fp64_avx512, f64x8, dummy_t, double, 8);
MIN_MAX_K1_KERNEL(max_k1_fp64_avx512, dummy_t, f64x8, double, 8);
MIN_MAX_KN_KERNEL(max_kn_fp64_avx512, dummy_t, f64x8, double, 8);

SUM_FLOAT_K1_KERNEL(sum_k1_fp64_avx512, double, double, 0, 1, identity);
SUM_FLOAT_KN_KERNEL(sum_kn_fp64_avx512, double, double, 8, identity);
SUM_FLOAT_K1_KERNEL(sum_k1_fp32_avx512, float, float, 0, 1, identity);
SUM_FLOAT_KN_KERNEL(sum_kn_fp32_avx512, float, float, 16, identity);
SUM_FLOAT_K1_KERNEL(sum_k1_bf16_fp32_avx512, bfloat16, float, 0, 2, identity);
SUM_FLOAT_KN_KERNEL(sum_kn_bf16_fp32_avx512, bfloat16, float, 16, identity);
SUM_FLOAT_K1_KERNEL(sum_k1_fp16_fp32_avx512, half, float, 0, 1, identity);
SUM_FLOAT_KN_KERNEL(sum_kn_fp16_fp32_avx512, half, float, 32, identity);
SUM_K1_KERNEL(sum_k1_int8_int32_avx512, int8_t, int32_t, 16, 4, identity);
SUM_KN_KERNEL(sum_kn_int8_int32_avx512, int8_t, int32_t, 64, identity);
SUM_K1_KERNEL(sum_k1_uint8_int32_avx512, uint8_t, int32_t, 16, 4, identity);
SUM_KN_KERNEL(sum_kn_uint8_int32_avx512, uint8_t, int32_t, 64, identity);
SUM_K1_KERNEL(sum_k1_int32_avx512, int32_t, int32_t, 16, 1, identity);
SUM_KN_KERNEL(sum_kn_int32_avx512, int32_t, int32_t, 16, identity);

SUM_FLOAT_K1_KERNEL(sum_squared_k1_fp64_avx512, double, double, 0, 1, square);
SUM_FLOAT_KN_KERNEL(sum_squared_kn_fp64_avx512, double, double, 8, square);
SUM_FLOAT_K1_KERNEL(sum_squared_k1_fp32_avx512, float, float, 0, 1, square);
SUM_FLOAT_KN_KERNEL(sum_squared_kn_fp32_avx512, float, float, 16, square);
SUM_FLOAT_K1_KERNEL(sum_squared_k1_bf16_fp32_avx512, bfloat16, float, 0, 2,
                    square);
SUM_FLOAT_KN_KERNEL(sum_squared_kn_bf16_fp32_avx512, bfloat16, float, 32,
                    square);
SUM_FLOAT_K1_KERNEL(sum_squared_k1_fp16_fp32_avx512, half, float, 0, 1, square);
SUM_FLOAT_KN_KERNEL(sum_squared_kn_fp16_fp32_avx512, half, float, 32, square);
SUM_K1_KERNEL(sum_squared_k1_int8_int32_avx512, int8_t, int32_t, 16, 4, square);
SUM_KN_KERNEL(sum_squared_kn_int8_int32_avx512, int8_t, int32_t, 16, square);
SUM_K1_KERNEL(sum_squared_k1_uint8_int32_avx512, uint8_t, int32_t, 16, 4,
              square);
SUM_KN_KERNEL(sum_squared_kn_uint8_int32_avx512, uint8_t, int32_t, 64, square);

}  // namespace ynn
