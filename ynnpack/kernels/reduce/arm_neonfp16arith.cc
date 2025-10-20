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
#include "ynnpack/kernels/reduce/min_max_accumulator.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/arm_neon_xf16.h"

namespace ynn {

namespace simd {

using f16x8x8 = multi_vec<f16x8, 8>;

YNN_ALWAYS_INLINE f32x4x2& operator+=(f32x4x2& a, f16x8 x) {
  f32x4 a_0(vcvt_f32_f16(vget_low_f16(reinterpret_cast<float16x8_t>(x.v))));
  f32x4 a_1(vcvt_f32_f16(vget_high_f16(reinterpret_cast<float16x8_t>(x.v))));

  a.v[0] += a_0;
  a.v[1] += a_1;

  return a;
}

static f32x4x16& operator+=(f32x4x16& a, f16x8x8 x) {
  YNN_UNROLL
  for (size_t i = 0; i < 8; ++i) {
    f32x4 b_0(vcvt_f32_f16(vget_low_f16(
        reinterpret_cast<float16x8_t>(x.v[i].v))));
    f32x4 b_1(vcvt_f32_f16(vget_high_f16(
        reinterpret_cast<float16x8_t>(x.v[i].v))));

    a.v[2 * i] += b_0;
    a.v[2 * i + 1] += b_1;
  }

  return a;
}

}  // namespace simd

using simd::f32x4x2;
using simd::f32x4x16;
using simd::f16x8x8;

void sum_fp16_fp32_neonfp16arith(size_t n, size_t k3, size_t k2, size_t k1,
                                 size_t a_stride_n, size_t a_stride_k3,
                                 size_t a_stride_k2, const void* a, size_t,
                                 void* c) {
  if (k1 == 1 && a_stride_n == sizeof(half)) {
    tiled_reduce<sum_accumulator_k1_1<f16x8x8, f32x4x16>, half, float>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const half*>(a),
        /*C_stride_m=*/0, reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<f32x4x2, 8>, half, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const half*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

}  // namespace ynn
