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

SUM_KERNEL(sum_bf16_fp32_neonbf16, f32x4, bfloat16, float, 8);

SUM_SQUARED_KERNEL(sum_squared_bf16_fp32_neonbf16, f32x4, bfloat16, float, 8);

}  // namespace ynn
