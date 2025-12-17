// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>

#include <cstddef>
#include <type_traits>

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/simd/arm_neonfma.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace simd {

static f32x8 reduce_add(
    f32x8 a, bf16x8 b, Square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  auto b_f32 = convert(b, float{});
  a = fma(b_f32, b_f32, a);
  return a;
}

static f32x4 reduce_add(
    f32x4 a, bf16x8 b, Square /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  uint32x4_t pairs = vreinterpretq_u32_u16(b.v);
  f32x4 even(vreinterpretq_f32_u32(vshlq_n_u32(pairs, 16)));
  f32x4 odd(vreinterpretq_f32_u32(vandq_u32(pairs, vdupq_n_u32(0xFFFF0000))));

  a = fma(odd, odd, a);
  a = fma(even, even, a);
  return a;
}

}  // namespace simd

using simd::bf16x8;
using simd::bf16x8;
using simd::f16x8;
using simd::f32x4;
using simd::f32x8;
using simd::f32x8;

void sum_squared_bf16_fp32_neonfma(size_t n, size_t k3, size_t k2, size_t k1,
                                   size_t a_stride_n, size_t a_stride_k3,
                                   size_t a_stride_k2, const void* a, size_t,
                                   void* c) {
  if (k1 == 1 && a_stride_n == sizeof(bfloat16)) {
    stream_reduce<sum_accumulator_k1_1<bf16x8, f32x8, Square>, bfloat16, float>(
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

void sum_squared_fp32_neonfma(size_t n, size_t k3, size_t k2, size_t k1,
                              size_t a_stride_n, size_t a_stride_k3,
                              size_t a_stride_k2, const void* a, size_t,
                              void* c) {
  if (k1 == 1 && a_stride_n == sizeof(float)) {
    stream_reduce<sum_accumulator_k1_1<f32x4, f32x4, Square>, float, float>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const float*>(a),
        /*C_stride_m=*/0, reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<f32x4, 4, Square>, float, float>(
      n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
      reinterpret_cast<const float*>(a), /*C_stride_m=*/0,
      reinterpret_cast<float*>(c));
  }
}

}  // namespace ynn
