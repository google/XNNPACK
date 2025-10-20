// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/simd/arm.h"
#include "ynnpack/base/simd/multi_vec.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace simd {

using s32x4x4 = multi_vec<s32x4, 4>;

static s32x4x4& operator+=(s32x4x4& a, s8x16 b) {
  int16x8_t b_lo = vmovl_s8(vget_low_s8(b.v));
  int16x8_t b_hi = vmovl_s8(vget_high_s8(b.v));

  s32x4 b_0(vmovl_s16(vget_low_s16(b_lo)));
  s32x4 b_1(vmovl_s16(vget_high_s16(b_lo)));
  s32x4 b_2(vmovl_s16(vget_low_s16(b_hi)));
  s32x4 b_3(vmovl_s16(vget_high_s16(b_hi)));

  a.v[0] += b_0;
  a.v[1] += b_1;
  a.v[2] += b_2;
  a.v[3] += b_3;

  return a;
}

static s32x4x4& operator+=(s32x4x4& a, u8x16 b) {
  int16x8_t b_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b.v)));
  int16x8_t b_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b.v)));

  s32x4 b_0(vmovl_s16(vget_low_s16(b_lo)));
  s32x4 b_1(vmovl_s16(vget_high_s16(b_lo)));
  s32x4 b_2(vmovl_s16(vget_low_s16(b_hi)));
  s32x4 b_3(vmovl_s16(vget_high_s16(b_hi)));

  a.v[0] += b_0;
  a.v[1] += b_1;
  a.v[2] += b_2;
  a.v[3] += b_3;

  return a;
}

static s32x4& reduce_add(
    s32x4& a, s8x16 b,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  a.v = vdotq_s32(a.v, b.v, vdupq_n_s8(1));
  return a;
}

// We want to accumulate uint8 dot products in int32 accumulators.
static s32x4& reduce_add(
    s32x4& a, u8x16 b,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  a.v = vreinterpretq_s32_u32(vdotq_u32(vreinterpretq_u32_s32(a.v), b.v,
                              vdupq_n_u8(1)));
  return a;
}

}  // namespace simd

using simd::s32x4;
using simd::s32x4x4;
using simd::s8x16;
using simd::u8x16;

void sum_int8_int32_neondot(size_t n, size_t k3, size_t k2, size_t k1,
                            size_t a_stride_n, size_t a_stride_k3,
                            size_t a_stride_k2, const void* a, size_t,
                            void* c) {
  if (k1 == 1 && a_stride_n == sizeof(int8_t)) {
    tiled_reduce<sum_accumulator_k1_1<s8x16, s32x4x4>, int8_t, int32_t>(
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
    tiled_reduce<sum_accumulator_k1_1<u8x16, s32x4x4>, uint8_t, int32_t>(
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

}  // namespace ynn
