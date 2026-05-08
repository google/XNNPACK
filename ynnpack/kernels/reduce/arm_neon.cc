// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/arm_neon.h"

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
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

}  // namespace simd

using simd::bf16x8;
using simd::f16x8;
using simd::f32x4;
using simd::f32x8;
using simd::s32x4;
using simd::s16x8;
using simd::s8x16;
using simd::u8x16;

using f16x8_rvar = float16_wrapper<f16x8, s16x8>;
using bf16x8_rvar = float16_wrapper<bf16x8, s16x8>;

MIN_MAX_K1_KERNEL(min_max_fp32_k1_neon, f32x4, f32x4, float, 4);
MIN_MAX_KN_KERNEL(min_max_fp32_kn_neon, f32x4, f32x4, float, 4);
MIN_MAX_K1_KERNEL(min_max_bf16_k1_neon, bf16x8_rvar, bf16x8_rvar, bfloat16, 8);
MIN_MAX_KN_KERNEL(min_max_bf16_kn_neon, bf16x8_rvar, bf16x8_rvar, bfloat16, 8);
MIN_MAX_K1_KERNEL(min_max_fp16_k1_neon, f16x8_rvar, f16x8_rvar, half, 8);
MIN_MAX_KN_KERNEL(min_max_fp16_kn_neon, f16x8_rvar, f16x8_rvar, half, 8);
MIN_MAX_K1_KERNEL(min_max_uint8_k1_neon, u8x16, u8x16, uint8_t, 16);
MIN_MAX_KN_KERNEL(min_max_uint8_kn_neon, u8x16, u8x16, uint8_t, 16);
MIN_MAX_K1_KERNEL(min_max_int8_k1_neon, s8x16, s8x16, int8_t, 16);
MIN_MAX_KN_KERNEL(min_max_int8_kn_neon, s8x16, s8x16, int8_t, 16);

MIN_MAX_K1_KERNEL(min_fp32_k1_neon, f32x4, dummy_t, float, 4);
MIN_MAX_KN_KERNEL(min_fp32_kn_neon, f32x4, dummy_t, float, 4);
MIN_MAX_K1_KERNEL(min_bf16_k1_neon, bf16x8_rvar, dummy_t, bfloat16, 8);
MIN_MAX_KN_KERNEL(min_bf16_kn_neon, bf16x8_rvar, dummy_t, bfloat16, 8);
MIN_MAX_K1_KERNEL(min_fp16_k1_neon, f16x8_rvar, dummy_t, half, 8);
MIN_MAX_KN_KERNEL(min_fp16_kn_neon, f16x8_rvar, dummy_t, half, 8);
MIN_MAX_K1_KERNEL(min_uint8_k1_neon, u8x16, dummy_t, uint8_t, 16);
MIN_MAX_KN_KERNEL(min_uint8_kn_neon, u8x16, dummy_t, uint8_t, 16);
MIN_MAX_K1_KERNEL(min_int8_k1_neon, s8x16, dummy_t, int8_t, 16);
MIN_MAX_KN_KERNEL(min_int8_kn_neon, s8x16, dummy_t, int8_t, 16);

MIN_MAX_K1_KERNEL(max_fp32_k1_neon, dummy_t, f32x4, float, 4);
MIN_MAX_KN_KERNEL(max_fp32_kn_neon, dummy_t, f32x4, float, 4);
MIN_MAX_K1_KERNEL(max_bf16_k1_neon, dummy_t, bf16x8_rvar, bfloat16, 8);
MIN_MAX_KN_KERNEL(max_bf16_kn_neon, dummy_t, bf16x8_rvar, bfloat16, 8);
MIN_MAX_K1_KERNEL(max_fp16_k1_neon, dummy_t, f16x8_rvar, half, 8);
MIN_MAX_KN_KERNEL(max_fp16_kn_neon, dummy_t, f16x8_rvar, half, 8);
MIN_MAX_K1_KERNEL(max_uint8_k1_neon, dummy_t, u8x16, uint8_t, 16);
MIN_MAX_KN_KERNEL(max_uint8_kn_neon, dummy_t, u8x16, uint8_t, 16);
MIN_MAX_K1_KERNEL(max_int8_k1_neon, dummy_t, s8x16, int8_t, 16);
MIN_MAX_KN_KERNEL(max_int8_kn_neon, dummy_t, s8x16, int8_t, 16);

SUM_K1_KERNEL(sum_bf16_fp32_k1_neon, bfloat16, float, 4, 2, identity);
SUM_KN_KERNEL(sum_bf16_fp32_kn_neon, bfloat16, float, 8, identity);
SUM_K1_KERNEL(sum_fp32_k1_neon, float, float, 4, 1, identity);
SUM_KN_KERNEL(sum_fp32_kn_neon, float, float, 4, identity);
SUM_K1_KERNEL(sum_int32_k1_neon, int32_t, int32_t, 4, 1, identity);
SUM_KN_KERNEL(sum_int32_kn_neon, int32_t, int32_t, 4, identity);

SUM_K1_KERNEL(sum_squared_bf16_fp32_k1_neon, bfloat16, float, 4, 2, square);
SUM_KN_KERNEL(sum_squared_bf16_fp32_kn_neon, bfloat16, float, 8, square);
SUM_K1_KERNEL(sum_squared_fp32_k1_neon, float, float, 4, 1, square);
SUM_KN_KERNEL(sum_squared_fp32_kn_neon, float, float, 4, square);

}  // namespace ynn
