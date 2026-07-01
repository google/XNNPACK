// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/simd/arm_vec128.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum.h"

namespace ynn {

namespace simd {

static f32x4 reduce_add(
    f32x4 a, bf16x8 b, identity /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  uint32x4_t pairs = vreinterpretq_u32_u16(b.v);
  f32x4 even(vreinterpretq_f32_u32(vshlq_n_u32(pairs, 16)));
  f32x4 odd(vreinterpretq_f32_u32(vandq_u32(pairs, vdupq_n_u32(0xFFFF0000))));

  a += odd;
  a += even;
  return a;
}

static f32x4 reduce_add(
    f32x4 a, bf16x8 b, square /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  uint32x4_t pairs = vreinterpretq_u32_u16(b.v);
  f32x4 even(vreinterpretq_f32_u32(vshlq_n_u32(pairs, 16)));
  f32x4 odd(vreinterpretq_f32_u32(vandq_u32(pairs, vdupq_n_u32(0xFFFF0000))));

  a += odd * odd;
  a += even * even;
  return a;
}

static s32x8 operator+(s32x8 a, s16x8 b) {
  a[0].v = vaddw_s16(a[0].v, vget_low_s16(b.v));
  a[1].v = vaddw_s16(a[1].v, vget_high_s16(b.v));
  return a;
}

static s32x8 operator+(s32x8 a, u16x8 b) {
  a[0].v = vreinterpretq_s32_u32(
      vaddw_u16(vreinterpretq_u32_s32(a[0].v), vget_low_u16(b.v)));
  a[1].v = vreinterpretq_s32_u32(
      vaddw_u16(vreinterpretq_u32_s32(a[1].v), vget_high_u16(b.v)));
  return a;
}

static s32x16 reduce_add(
    s32x16 a, s8x16 b, square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  int8x8_t b_lo_s8 = vget_low_s8(b.v);
  int8x8_t b_hi_s8 = vget_high_s8(b.v);
  s16x8 sq_lo{vmull_s8(b_lo_s8, b_lo_s8)};
  s16x8 sq_hi{vmull_s8(b_hi_s8, b_hi_s8)};

  return concat(extract<0>(a, s32x8::N) + sq_lo,
                extract<1>(a, s32x8::N) + sq_hi);
}

static s32x16 reduce_add(
    s32x16 a, u8x16 b, square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  uint8x8_t b_lo_s8 = vget_low_u8(b.v);
  uint8x8_t b_hi_s8 = vget_high_u8(b.v);
  u16x8 sq_lo{vmull_u8(b_lo_s8, b_lo_s8)};
  u16x8 sq_hi{vmull_u8(b_hi_s8, b_hi_s8)};

  return concat(extract<0>(a, s32x8::N) + sq_lo,
                extract<1>(a, s32x8::N) + sq_hi);
}

}  // namespace simd

using simd::bf16x8;
using simd::f16x8;
using simd::f32x4;
using simd::f32x8;
using simd::s16x8;
using simd::s32x4;
using simd::s8x16;
using simd::u8x16;

using xf16x8 = sign_magnitude<s16x8>;
using xf8x16 = sign_magnitude<s8x16>;

MIN_MAX_K1_KERNEL(min_max_k1_fp32_neon, f32x4, f32x4, float, 4);
MIN_MAX_KN_KERNEL(min_max_kn_fp32_neon, f32x4, f32x4, float, 4);
MIN_MAX_K1_KERNEL(min_max_k1_xf16_neon, xf16x8, xf16x8, int16_t, 8);
MIN_MAX_KN_KERNEL(min_max_kn_xf16_neon, xf16x8, xf16x8, int16_t, 8);
MIN_MAX_K1_KERNEL(min_max_k1_xf8_neon, xf8x16, xf8x16, int8_t, 16);
MIN_MAX_KN_KERNEL(min_max_kn_xf8_neon, xf8x16, xf8x16, int8_t, 16);
MIN_MAX_K1_KERNEL(min_max_k1_uint8_neon, u8x16, u8x16, uint8_t, 16);
MIN_MAX_KN_KERNEL(min_max_kn_uint8_neon, u8x16, u8x16, uint8_t, 16);
MIN_MAX_K1_KERNEL(min_max_k1_int8_neon, s8x16, s8x16, int8_t, 16);
MIN_MAX_KN_KERNEL(min_max_kn_int8_neon, s8x16, s8x16, int8_t, 16);

MIN_MAX_K1_KERNEL(min_k1_fp32_neon, f32x4, dummy_t, float, 4);
MIN_MAX_KN_KERNEL(min_kn_fp32_neon, f32x4, dummy_t, float, 4);
MIN_MAX_K1_KERNEL(min_k1_xf16_neon, xf16x8, dummy_t, int16_t, 8);
MIN_MAX_KN_KERNEL(min_kn_xf16_neon, xf16x8, dummy_t, int16_t, 8);
MIN_MAX_K1_KERNEL(min_k1_xf8_neon, xf8x16, dummy_t, int8_t, 16);
MIN_MAX_KN_KERNEL(min_kn_xf8_neon, xf8x16, dummy_t, int8_t, 16);
MIN_MAX_K1_KERNEL(min_k1_uint8_neon, u8x16, dummy_t, uint8_t, 16);
MIN_MAX_KN_KERNEL(min_kn_uint8_neon, u8x16, dummy_t, uint8_t, 16);
MIN_MAX_K1_KERNEL(min_k1_int8_neon, s8x16, dummy_t, int8_t, 16);
MIN_MAX_KN_KERNEL(min_kn_int8_neon, s8x16, dummy_t, int8_t, 16);

MIN_MAX_K1_KERNEL(max_k1_fp32_neon, dummy_t, f32x4, float, 4);
MIN_MAX_KN_KERNEL(max_kn_fp32_neon, dummy_t, f32x4, float, 4);
MIN_MAX_K1_KERNEL(max_k1_xf16_neon, dummy_t, xf16x8, int16_t, 8);
MIN_MAX_KN_KERNEL(max_kn_xf16_neon, dummy_t, xf16x8, int16_t, 8);
MIN_MAX_K1_KERNEL(max_k1_xf8_neon, dummy_t, xf8x16, int8_t, 16);
MIN_MAX_KN_KERNEL(max_kn_xf8_neon, dummy_t, xf8x16, int8_t, 16);
MIN_MAX_K1_KERNEL(max_k1_uint8_neon, dummy_t, u8x16, uint8_t, 16);
MIN_MAX_KN_KERNEL(max_kn_uint8_neon, dummy_t, u8x16, uint8_t, 16);
MIN_MAX_K1_KERNEL(max_k1_int8_neon, dummy_t, s8x16, int8_t, 16);
MIN_MAX_KN_KERNEL(max_kn_int8_neon, dummy_t, s8x16, int8_t, 16);

SUM_FLOAT_K1_KERNEL(sum_k1_bf16_fp32_neon, bfloat16, float, 4, 2, identity);
SUM_FLOAT_KN_KERNEL(sum_kn_bf16_fp32_neon, bfloat16, float, 8, identity);
SUM_FLOAT_K1_KERNEL(sum_k1_fp32_neon, float, float, 4, 1, identity);
SUM_FLOAT_KN_KERNEL(sum_kn_fp32_neon, float, float, 4, identity);
SUM_K1_KERNEL(sum_k1_int32_neon, int32_t, int32_t, 4, 1, identity);
SUM_KN_KERNEL(sum_kn_int32_neon, int32_t, int32_t, 4, identity);
SUM_KN_KERNEL(sum_kn_uint8_int32_neon, uint8_t, int32_t, 16, identity);
SUM_KN_KERNEL(sum_kn_int8_int32_neon, int8_t, int32_t, 16, identity);

SUM_FLOAT_K1_KERNEL(sum_squared_k1_bf16_fp32_neon, bfloat16, float, 4, 2,
                    square);
SUM_FLOAT_KN_KERNEL(sum_squared_kn_bf16_fp32_neon, bfloat16, float, 8, square);
SUM_FLOAT_K1_KERNEL(sum_squared_k1_fp32_neon, float, float, 4, 1, square);
SUM_FLOAT_KN_KERNEL(sum_squared_kn_fp32_neon, float, float, 4, square);
SUM_KN_KERNEL(sum_squared_kn_uint8_int32_neon, uint8_t, int32_t, 16, square);
SUM_KN_KERNEL(sum_squared_kn_int8_int32_neon, int8_t, int32_t, 16, square);

}  // namespace ynn
