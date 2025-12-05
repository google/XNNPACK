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
#include "ynnpack/base/simd/arm_neon.h"
#include "ynnpack/base/simd/multi_vec.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max_accumulator.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace simd {

using bf16x8x8 = multi_vec<bf16x8, 8>;
using f32x4x16 = multi_vec<f32x4, 16>;

// BFMLAL is not used here for consistency - it produces different results from
// BFDOT.
static f32x4x16& operator+=(f32x4x16& a, bf16x8x8 b) {
  bfloat16x8_t one = vreinterpretq_bf16_u16(vdupq_n_u16(0x3F80));
  uint16x8_t zero = vdupq_n_u16(0x0000);

  YNN_UNROLL
  for (size_t i = 0; i < 8; ++i) {
    uint16x8x2_t zipped = vzipq_u16(b.v[i].v, zero);
    a.v[2 * i].v = vbfdotq_f32(a.v[2 * i].v,
        vreinterpretq_bf16_u16(zipped.val[0]), one);
    a.v[2 * i + 1].v = vbfdotq_f32(a.v[2 * i + 1].v,
        vreinterpretq_bf16_u16(zipped.val[1]), one);
  }

  return a;
}

static f32x4 reduce_add(
    f32x4 a, bf16x8 b, Identity /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  return f32x4{vbfdotq_f32(a.v, vreinterpretq_bf16_u16(b.v),
      vreinterpretq_bf16_u16(vdupq_n_u16(0x3F80)))};
}

// BFMLAL is not used here for consistency - it produces different results from
// BFDOT.
static f32x4x16 reduce_add(
    f32x4x16 a, bf16x8x8 b, Square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  uint16x8_t zero = vdupq_n_u16(0x0000);

  YNN_UNROLL
  for (size_t i = 0; i < 8; ++i) {
    uint16x8x2_t zipped = vzipq_u16(b.v[i].v, zero);
    a.v[2 * i].v = vbfdotq_f32(a.v[2 * i].v,
        vreinterpretq_bf16_u16(zipped.val[0]),
        vreinterpretq_bf16_u16(zipped.val[0]));
    a.v[2 * i + 1].v = vbfdotq_f32(a.v[2 * i + 1].v,
        vreinterpretq_bf16_u16(zipped.val[1]),
        vreinterpretq_bf16_u16(zipped.val[1]));
  }

  return a;
}

static f32x4 reduce_add(
    f32x4 a, bf16x8 b, Square /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  return f32x4{vbfdotq_f32(a.v, vreinterpretq_bf16_u16(b.v),
      vreinterpretq_bf16_u16(b.v))};
}

}  // namespace simd

using simd::f32x4;
using simd::f32x4x16;
using simd::bf16x8x8;

void sum_bf16_fp32_neonbf16(size_t n, size_t k3, size_t k2, size_t k1,
                            size_t a_stride_n, size_t a_stride_k3,
                            size_t a_stride_k2, const void* a, size_t,
                            void* c) {
  if (k1 == 1 && a_stride_n == sizeof(bfloat16)) {
    tiled_reduce<sum_accumulator_k1_1<bf16x8x8, f32x4x16>, bfloat16, float>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<f32x4, 8>, bfloat16, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

void sum_squared_bf16_fp32_neonbf16(size_t n, size_t k3, size_t k2, size_t k1,
                                    size_t a_stride_n, size_t a_stride_k3,
                                    size_t a_stride_k2, const void* a, size_t,
                                    void* c) {
  if (k1 == 1 && a_stride_n == sizeof(bfloat16)) {
    tiled_reduce<sum_accumulator_k1_1<bf16x8x8, f32x4x16, Square>, bfloat16,
      float>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<f32x4, 8, Square>, bfloat16, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

}  // namespace ynn
