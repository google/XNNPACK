// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/simd/arm_vec128.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum.h"

namespace ynn {

namespace simd {

static s32x4 reduce_add(
    s32x4 a, s8x16 b, identity /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  a.v = vdotq_s32(a.v, b.v, vdupq_n_s8(1));
  return a;
}

// We want to accumulate uint8 dot products in int32 accumulators.
static s32x4 reduce_add(
    s32x4 a, u8x16 b, identity /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  a.v = vreinterpretq_s32_u32(vdotq_u32(vreinterpretq_u32_s32(a.v), b.v,
                              vdupq_n_u8(1)));
  return a;
}

static s32x4 reduce_add(
    s32x4 a, s8x16 b, square /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  a.v = vdotq_s32(a.v, b.v, b.v);
  return a;
}

// We want to accumulate uint8 dot products in int32 accumulators.
static s32x4 reduce_add(
    s32x4 a, u8x16 b, square /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  a.v = vreinterpretq_s32_u32(vdotq_u32(vreinterpretq_u32_s32(a.v), b.v, b.v));
  return a;
}

}  // namespace simd

using simd::s32x4;
using simd::s32x16;
using simd::s8x16;
using simd::u8x16;

SUM_K1_KERNEL(sum_k1_uint8_int32_neondot, uint8_t, int32_t, 4, 4, identity);
SUM_K1_KERNEL(sum_k1_int8_int32_neondot, int8_t, int32_t, 4, 4, identity);

SUM_K1_KERNEL(sum_squared_k1_uint8_int32_neondot, uint8_t, int32_t, 4, 4,
              square);
SUM_K1_KERNEL(sum_squared_k1_int8_int32_neondot, int8_t, int32_t, 4, 4, square);

}  // namespace ynn
