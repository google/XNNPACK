// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/hexagon_hvx.h"

#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max.h"
#include "ynnpack/kernels/reduce/sum.h"

namespace ynn {

namespace simd {

static s32x32 reduce_add(
    s32x32 a, u8x128 b, identity /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  a.v = Q6_Vw_vrmpyacc_VwVubRb(a.v, b.v, 0x01010101);
  return a;
}

static s32x32 reduce_add(
    s32x32 a, u8x128 b, square /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  a.v = Q6_Vuw_vrmpyacc_VuwVubVub(a.v, b.v, b.v);
  return a;
}

static s32x32 reduce_add(
    s32x32 a, s8x128 b, identity /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  const auto ones = Q6_V_vsplat_R(0x01010101);
  a.v = Q6_Vw_vrmpyacc_VwVbVb(a.v, b.v, ones);
  return a;
}

static s32x32 reduce_add(
    s32x32 a, s8x128 b, square /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  a.v = Q6_Vw_vrmpyacc_VwVbVb(a.v, b.v, b.v);
  return a;
}

static f32x32 reduce_add(
    f32x32 a, f32x32 b, identity /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  return a + b;
}

static f32x32 reduce_add(
    f32x32 a, f32x32 b, square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  return a + (b * b);
}

}  // namespace simd

using simd::bf16x64;
using simd::f16x64;
using simd::f32x32;
using simd::s16x64;
using simd::s32x32;
using simd::s8x128;
using simd::u8x128;
using s32x128 = simd::vec<int32_t, 128>;
using f32x128 = simd::vec<float, 128>;

using bf16x64_rvar = float16_wrapper<bf16x64, s16x64>;

MIN_MAX_K1_KERNEL(min_max_k1_fp32_hvx, f32x32, f32x32, float, 32);
MIN_MAX_KN_KERNEL(min_max_kn_fp32_hvx, f32x32, f32x32, float, 32);
MIN_MAX_K1_KERNEL(min_max_k1_fp16_hvx, f16x64, f16x64, half, 64);
MIN_MAX_KN_KERNEL(min_max_kn_fp16_hvx, f16x64, f16x64, half, 64);
MIN_MAX_K1_KERNEL(min_max_k1_bf16_hvx, bf16x64_rvar, bf16x64_rvar, bfloat16,
                  64);
MIN_MAX_KN_KERNEL(min_max_kn_bf16_hvx, bf16x64_rvar, bf16x64_rvar, bfloat16,
                  64);
MIN_MAX_K1_KERNEL(min_max_k1_uint8_hvx, u8x128, u8x128, uint8_t, 128);
MIN_MAX_KN_KERNEL(min_max_kn_uint8_hvx, u8x128, u8x128, uint8_t, 128);
MIN_MAX_K1_KERNEL(min_max_k1_int8_hvx, s8x128, s8x128, int8_t, 128);
MIN_MAX_KN_KERNEL(min_max_kn_int8_hvx, s8x128, s8x128, int8_t, 128);

MIN_MAX_K1_KERNEL(min_k1_fp32_hvx, f32x32, dummy_t, float, 32);
MIN_MAX_KN_KERNEL(min_kn_fp32_hvx, f32x32, dummy_t, float, 32);
MIN_MAX_K1_KERNEL(min_k1_fp16_hvx, f16x64, dummy_t, half, 64);
MIN_MAX_KN_KERNEL(min_kn_fp16_hvx, f16x64, dummy_t, half, 64);
MIN_MAX_K1_KERNEL(min_k1_bf16_hvx, bf16x64_rvar, dummy_t, bfloat16, 64);
MIN_MAX_KN_KERNEL(min_kn_bf16_hvx, bf16x64_rvar, dummy_t, bfloat16, 64);
MIN_MAX_K1_KERNEL(min_k1_uint8_hvx, u8x128, dummy_t, uint8_t, 128);
MIN_MAX_KN_KERNEL(min_kn_uint8_hvx, u8x128, dummy_t, uint8_t, 128);
MIN_MAX_K1_KERNEL(min_k1_int8_hvx, s8x128, dummy_t, int8_t, 128);
MIN_MAX_KN_KERNEL(min_kn_int8_hvx, s8x128, dummy_t, int8_t, 128);

MIN_MAX_K1_KERNEL(max_k1_fp32_hvx, dummy_t, f32x32, float, 32);
MIN_MAX_KN_KERNEL(max_kn_fp32_hvx, dummy_t, f32x32, float, 32);
MIN_MAX_K1_KERNEL(max_k1_fp16_hvx, dummy_t, f16x64, half, 64);
MIN_MAX_KN_KERNEL(max_kn_fp16_hvx, dummy_t, f16x64, half, 64);
MIN_MAX_K1_KERNEL(max_k1_bf16_hvx, dummy_t, bf16x64_rvar, bfloat16, 64);
MIN_MAX_KN_KERNEL(max_kn_bf16_hvx, dummy_t, bf16x64_rvar, bfloat16, 64);
MIN_MAX_K1_KERNEL(max_k1_uint8_hvx, dummy_t, u8x128, uint8_t, 128);
MIN_MAX_KN_KERNEL(max_kn_uint8_hvx, dummy_t, u8x128, uint8_t, 128);
MIN_MAX_K1_KERNEL(max_k1_int8_hvx, dummy_t, s8x128, int8_t, 128);
MIN_MAX_KN_KERNEL(max_kn_int8_hvx, dummy_t, s8x128, int8_t, 128);

SUM_K1_KERNEL(sum_k1_uint8_int32_hvx, uint8_t, int32_t, 32, 4, identity);
SUM_KN_KERNEL(sum_kn_uint8_int32_hvx, uint8_t, int32_t, 128, identity);
SUM_K1_KERNEL(sum_k1_int8_int32_hvx, int8_t, int32_t, 32, 4, identity);
SUM_KN_KERNEL(sum_kn_int8_int32_hvx, int8_t, int32_t, 128, identity);

SUM_K1_KERNEL(sum_squared_k1_uint8_int32_hvx, uint8_t, int32_t, 32, 4, square);
SUM_KN_KERNEL(sum_squared_kn_uint8_int32_hvx, uint8_t, int32_t, 128, square);
SUM_K1_KERNEL(sum_squared_k1_int8_int32_hvx, int8_t, int32_t, 32, 4, square);
SUM_KN_KERNEL(sum_squared_kn_int8_int32_hvx, int8_t, int32_t, 128, square);

SUM_FLOAT_K1_KERNEL(sum_k1_fp32_hvx, float, float, 32, 1, identity);
SUM_FLOAT_KN_KERNEL(sum_kn_fp32_hvx, float, float, 32, identity);
SUM_FLOAT_K1_KERNEL(sum_squared_k1_fp32_hvx, float, float, 32, 1, square);
SUM_FLOAT_KN_KERNEL(sum_squared_kn_fp32_hvx, float, float, 32, square);

}  // namespace ynn
