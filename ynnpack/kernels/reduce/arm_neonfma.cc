// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/arm_neonfma.h"

#include <arm_neon.h>

#include <cstddef>
#include <type_traits>

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum.h"

namespace ynn {

namespace simd {

static f32x8 reduce_add(
    f32x8 a, bf16x8 b, square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  auto b_f32 = cast(b, float{});
  a = fma(b_f32, b_f32, a);
  return a;
}

static f32x4 reduce_add(
    f32x4 a, bf16x8 b, square /*map_fn*/,
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

SUM_FLOAT_K1_KERNEL(sum_squared_k1_bf16_fp32_neonfma, bfloat16, float, 8, 1,
                    square);
SUM_FLOAT_KN_KERNEL(sum_squared_kn_bf16_fp32_neonfma, bfloat16, float, 8,
                    square);
SUM_FLOAT_K1_KERNEL(sum_squared_k1_fp32_neonfma, float, float, 4, 1, square);
SUM_FLOAT_KN_KERNEL(sum_squared_kn_fp32_neonfma, float, float, 4, square);

}  // namespace ynn
