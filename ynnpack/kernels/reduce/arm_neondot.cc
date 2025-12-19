// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/simd/arm_neon.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace simd {

static s32x4 reduce_add(
    s32x4 a, s8x16 b, Identity /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  a.v = vdotq_s32(a.v, b.v, vdupq_n_s8(1));
  return a;
}

// We want to accumulate uint8 dot products in int32 accumulators.
static s32x4 reduce_add(
    s32x4 a, u8x16 b, Identity /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  a.v = vreinterpretq_s32_u32(vdotq_u32(vreinterpretq_u32_s32(a.v), b.v,
                              vdupq_n_u8(1)));
  return a;
}

static s32x16 reduce_add(
    s32x16 a, s8x16 b, Square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  int8x8_t b_lo_s8 = vget_low_s8(b.v);
  int8x8_t b_hi_s8 = vget_high_s8(b.v);
  int16x8_t sq_lo = vmull_s8(b_lo_s8, b_lo_s8);
  int16x8_t sq_hi = vmull_s8(b_hi_s8, b_hi_s8);

  a[0][0].v = vaddw_s16(a[0][0].v, vget_low_s16(sq_lo));
  a[0][1].v = vaddw_s16(a[0][1].v, vget_high_s16(sq_lo));
  a[1][0].v = vaddw_s16(a[1][0].v, vget_low_s16(sq_hi));
  a[1][1].v = vaddw_s16(a[1][1].v, vget_high_s16(sq_hi));

  return a;
}

static s32x16 reduce_add(
    s32x16 a, u8x16 b, Square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  uint8x8_t b_lo_s8 = vget_low_u8(b.v);
  uint8x8_t b_hi_s8 = vget_high_u8(b.v);
  uint16x8_t sq_lo = vmull_u8(b_lo_s8, b_lo_s8);
  uint16x8_t sq_hi = vmull_u8(b_hi_s8, b_hi_s8);

  a[0][0].v = vreinterpretq_s32_u32(
      vaddw_u16(vreinterpretq_u32_s32(a[0][0].v), vget_low_u16(sq_lo)));
  a[0][1].v = vreinterpretq_s32_u32(
      vaddw_u16(vreinterpretq_u32_s32(a[0][1].v), vget_high_u16(sq_lo)));
  a[1][0].v = vreinterpretq_s32_u32(
      vaddw_u16(vreinterpretq_u32_s32(a[1][0].v), vget_low_u16(sq_hi)));
  a[1][1].v = vreinterpretq_s32_u32(
      vaddw_u16(vreinterpretq_u32_s32(a[1][1].v), vget_high_u16(sq_hi)));

  return a;
}

static s32x4 reduce_add(
    s32x4 a, s8x16 b, Square /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  a.v = vdotq_s32(a.v, b.v, b.v);
  return a;
}

// We want to accumulate uint8 dot products in int32 accumulators.
static s32x4 reduce_add(
    s32x4 a, u8x16 b, Square /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  a.v = vreinterpretq_s32_u32(vdotq_u32(vreinterpretq_u32_s32(a.v), b.v, b.v));
  return a;
}

}  // namespace simd

using simd::s32x4;
using simd::s32x16;
using simd::s8x16;
using simd::u8x16;

void sum_int8_int32_neondot(size_t n, size_t k3, size_t k2, size_t k1,
                            size_t a_stride_n, size_t a_stride_k3,
                            size_t a_stride_k2, const void* a, size_t,
                            void* c) {
  if (k1 == 1 && a_stride_n == sizeof(int8_t)) {
    stream_reduce<sum_accumulator_k1_1<s32x16>, int8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const int8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x4, 16>, int8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const int8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

void sum_uint8_int32_neondot(size_t n, size_t k3, size_t k2, size_t k1,
                             size_t a_stride_n, size_t a_stride_k3,
                             size_t a_stride_k2, const void* a, size_t,
                             void* c) {
  if (k1 == 1 && a_stride_n == sizeof(uint8_t)) {
    stream_reduce<sum_accumulator_k1_1<s32x16>, uint8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x4, 16>, uint8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

void sum_squared_int8_int32_neondot(size_t n, size_t k3, size_t k2, size_t k1,
                                    size_t a_stride_n, size_t a_stride_k3,
                                    size_t a_stride_k2, const void* a, size_t,
                                    void* c) {
  if (k1 == 1 && a_stride_n == sizeof(int8_t)) {
    stream_reduce<sum_accumulator_k1_1<s32x16, Square>, int8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const int8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x4, 16, Square>, int8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const int8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

void sum_squared_uint8_int32_neondot(size_t n, size_t k3, size_t k2, size_t k1,
                                     size_t a_stride_n, size_t a_stride_k3,
                                     size_t a_stride_k2, const void* a, size_t,
                                     void* c) {
  if (k1 == 1 && a_stride_n == sizeof(uint8_t)) {
    stream_reduce<sum_accumulator_k1_1<s32x16, Square>, uint8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x4, 16, Square>, uint8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

}  // namespace ynn
