// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>

#include <cstddef>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/simd/arm_neon.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max_accumulator.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace simd {

static f32x8 reduce_add(
    f32x8 a, bf16x8 b, Identity /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  bfloat16x8_t one = vreinterpretq_bf16_u16(vdupq_n_u16(0x3F80));
  uint16x8x2_t zipped = vzipq_u16(b.v, vdupq_n_u16(0x0000));
  bfloat16x8_t evens = vreinterpretq_bf16_u16(zipped.val[0]);
  bfloat16x8_t odds = vreinterpretq_bf16_u16(zipped.val[1]);
  return concat(f32x4{vbfdotq_f32(extract<0>(a, f32x4::N).v, evens, one)},
                f32x4{vbfdotq_f32(extract<1>(a, f32x4::N).v, odds, one)});
}

static f32x4 reduce_add(
    f32x4 a, bf16x8 b, Identity /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  return f32x4{vbfdotq_f32(a.v, vreinterpretq_bf16_u16(b.v),
                           vreinterpretq_bf16_u16(vdupq_n_u16(0x3F80)))};
}

static f32x8 reduce_add(
    f32x8 a, bf16x8 b, Square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  uint16x8x2_t zipped = vzipq_u16(b.v, vdupq_n_u16(0x0000));
  bfloat16x8_t evens = vreinterpretq_bf16_u16(zipped.val[0]);
  bfloat16x8_t odds = vreinterpretq_bf16_u16(zipped.val[1]);
  return concat(f32x4{vbfdotq_f32(extract<0>(a, f32x4::N).v, evens, evens)},
                f32x4{vbfdotq_f32(extract<1>(a, f32x4::N).v, odds, odds)});
}

static f32x4 reduce_add(
    f32x4 a, bf16x8 b, Square /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  return f32x4{vbfdotq_f32(a.v, vreinterpretq_bf16_u16(b.v),
                           vreinterpretq_bf16_u16(b.v))};
}

}  // namespace simd

using simd::bf16x8;
using simd::f32x4;
using simd::f32x8;

void sum_bf16_fp32_neonbf16(size_t n, size_t k3, size_t k2, size_t k1,
                            size_t a_stride_n, size_t a_stride_k3,
                            size_t a_stride_k2, const void* a, size_t,
                            void* c) {
  if (k1 == 1 && a_stride_n == sizeof(bfloat16)) {
    stream_reduce<sum_accumulator_k1_1<f32x8>, bfloat16, float>(
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
    stream_reduce<sum_accumulator_k1_1<f32x8, Square>, bfloat16, float>(
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
