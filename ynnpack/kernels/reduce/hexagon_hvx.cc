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
#include "ynnpack/kernels/reduce/min_max_accumulator.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace simd {

static s32x32 reduce_add(
    s32x32 a, u8x128 b, Identity /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  a.v = Q6_Vw_vrmpyacc_VwVubRb(a.v, b.v, 0x01010101);
  return a;
}

static s32x32 reduce_add(
    s32x32 a, u8x128 b, Square /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  a.v = Q6_Vuw_vrmpyacc_VuwVubVub(a.v, b.v, b.v);
  return a;
}

static s32x32 reduce_add(
    s32x32 a, s8x128 b, Identity /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  const auto ones = Q6_V_vsplat_R(0x01010101);
  a.v = Q6_Vw_vrmpyacc_VwVbVb(a.v, b.v, ones);
  return a;
}

static s32x32 reduce_add(
    s32x32 a, s8x128 b, Square /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  a.v = Q6_Vw_vrmpyacc_VwVbVb(a.v, b.v, b.v);
  return a;
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

MIN_MAX_KERNEL(min_max_fp32_4x32_hvx, f32x32, f32x32, float, 32);
MIN_MAX_KERNEL(min_max_fp16_4x64_hvx, f16x64, f16x64, half, 64);
MIN_MAX_KERNEL(min_max_bf16_4x64_hvx, bf16x64_rvar, bf16x64_rvar, bfloat16, 64);
MIN_MAX_KERNEL(min_max_uint8_4x128_hvx, u8x128, u8x128, uint8_t, 128);
MIN_MAX_KERNEL(min_max_int8_4x128_hvx, s8x128, s8x128, int8_t, 128);

MIN_MAX_KERNEL(min_fp32_4x32_hvx, f32x32, dummy_t, float, 32);
MIN_MAX_KERNEL(min_fp16_4x64_hvx, f16x64, dummy_t, half, 64);
MIN_MAX_KERNEL(min_bf16_4x64_hvx, bf16x64_rvar, dummy_t, bfloat16, 64);
MIN_MAX_KERNEL(min_uint8_4x128_hvx, u8x128, dummy_t, uint8_t, 128);
MIN_MAX_KERNEL(min_int8_4x128_hvx, s8x128, dummy_t, int8_t, 128);

MIN_MAX_KERNEL(max_fp32_4x32_hvx, dummy_t, f32x32, float, 32);
MIN_MAX_KERNEL(max_fp16_4x64_hvx, dummy_t, f16x64, half, 64);
MIN_MAX_KERNEL(max_bf16_4x64_hvx, dummy_t, bf16x64_rvar, bfloat16, 64);
MIN_MAX_KERNEL(max_uint8_4x128_hvx, dummy_t, u8x128, uint8_t, 128);
MIN_MAX_KERNEL(max_int8_4x128_hvx, dummy_t, s8x128, int8_t, 128);

void sum_uint8_int32_hvx(size_t n, size_t k3, size_t k2, size_t k1,
                         size_t a_stride_n, size_t a_stride_k3,
                         size_t a_stride_k2, const void* a, size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(uint8_t)) {
    // TODO(b/482435301): This case is poorly optimized. It naively converts to
    // int32 and does a 32-bit add. We should be using a widening op, and
    // storing the accumulators interleaved until `sum_rows`.
    stream_reduce<sum_accumulator_k1_1<s32x128>, uint8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x32, 128, Identity>, uint8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

void sum_squared_uint8_int32_hvx(size_t n, size_t k3, size_t k2, size_t k1,
                                 size_t a_stride_n, size_t a_stride_k3,
                                 size_t a_stride_k2, const void* a, size_t,
                                 void* c) {
  if (k1 == 1 && a_stride_n == sizeof(uint8_t)) {
    // TODO(b/482435301): This case is poorly optimized. It naively converts to
    // int32 and does a 32-bit add. We should be using a widening op, and
    // storing the accumulators interleaved until `sum_rows`.
    stream_reduce<sum_accumulator_k1_1<s32x128, Square>, uint8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x32, 128, Square>, uint8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

void sum_int8_int32_hvx(size_t n, size_t k3, size_t k2, size_t k1,
                        size_t a_stride_n, size_t a_stride_k3,
                        size_t a_stride_k2, const void* a, size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(int8_t)) {
    // TODO(b/482435301): This case is poorly optimized. It naively converts to
    // int32 and does a 32-bit add. We should be using a widening op, and
    // storing the accumulators interleaved until `sum_rows`.
    stream_reduce<sum_accumulator_k1_1<s32x128>, int8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const int8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x32, 128, Identity>, int8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const int8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

void sum_squared_int8_int32_hvx(size_t n, size_t k3, size_t k2, size_t k1,
                                size_t a_stride_n, size_t a_stride_k3,
                                size_t a_stride_k2, const void* a, size_t,
                                void* c) {
  if (k1 == 1 && a_stride_n == sizeof(int8_t)) {
    // TODO(b/482435301): This case is poorly optimized. It naively converts to
    // int32 and does a 32-bit add. We should be using a widening op, and
    // storing the accumulators interleaved until `sum_rows`.
    stream_reduce<sum_accumulator_k1_1<s32x128, Square>, int8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const int8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x32, 128, Square>, int8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const int8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

}  // namespace ynn
