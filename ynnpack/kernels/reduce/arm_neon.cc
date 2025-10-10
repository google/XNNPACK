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
#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/arm.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max_accumulator.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"
#include "ynnpack/kernels/reduce/reduce.h"

namespace ynn {

using simd::bf16x8;
using simd::f16x8;
using simd::f32x4;
using simd::s16x8;
using simd::s32x4;
using simd::s8x16;
using simd::u8x16;

using f16x8_rvar = float16_wrapper<f16x8, s16x8>;
using bf16x8_rvar = float16_wrapper<bf16x8, s16x8>;

namespace {

struct accumulator_fp32 {
  static constexpr std::integral_constant<size_t, 4> N = {};
  static constexpr std::integral_constant<size_t, 4> K = {};

  f32x4 acc[N];

  accumulator_fp32() = default;

  YNN_ALWAYS_INLINE explicit accumulator_fp32(size_t) {
    for (size_t i = 0; i < N; ++i) {
      acc[i] = 0;
    }
  }

  template <typename NT, typename KT>
  YNN_ALWAYS_INLINE void reduce(const float* A, size_t A_stride_n, NT n, KT k) {
    const simd::vec<float, K> zero(0.0f);
    auto a_0 = load(offset_bytes(A, 0 * A_stride_n), zero, k);
    auto a_1 = 1 < n ? load(offset_bytes(A, 1 * A_stride_n), zero, k) : zero;
    auto a_2 = 2 < n ? load(offset_bytes(A, 2 * A_stride_n), zero, k) : zero;
    auto a_3 = 3 < n ? load(offset_bytes(A, 3 * A_stride_n), zero, k) : zero;
    acc[0] = acc[0] + a_0;
    acc[1] = acc[1] + a_1;
    acc[2] = acc[2] + a_2;
    acc[3] = acc[3] + a_3;
  }

  template <typename NT>
  YNN_ALWAYS_INLINE void accumulate(size_t /*C_stride_m*/, float* __restrict C,
                                    NT n) {
    std::array<f32x4, 4> acc_t =
        simd::transpose<float>({{acc[0], acc[1], acc[2], acc[3]}});
    f32x4 sum = (acc_t[0] + acc_t[1]) + (acc_t[2] + acc_t[3]);
    store(C, load(C, f32x4{}, n) + sum, n);
  }
};

}  // namespace

MIN_MAX_KERNEL(min_max_fp32_4x4_neon, f32x4, f32x4, float, 4);
MIN_MAX_KERNEL(min_max_bf16_4x8_neon, bf16x8_rvar, bf16x8_rvar, bfloat16, 8);
MIN_MAX_KERNEL(min_max_fp16_4x8_neon, f16x8_rvar, f16x8_rvar, half, 8);
MIN_MAX_KERNEL(min_max_uint8_4x16_neon, u8x16, u8x16, uint8_t, 16);
MIN_MAX_KERNEL(min_max_int8_4x16_neon, s8x16, s8x16, int8_t, 16);

MIN_MAX_KERNEL(min_fp32_4x4_neon, f32x4, dummy_t, float, 4);
MIN_MAX_KERNEL(min_bf16_4x8_neon, bf16x8_rvar, dummy_t, bfloat16, 8);
MIN_MAX_KERNEL(min_fp16_4x8_neon, f16x8_rvar, dummy_t, half, 8);
MIN_MAX_KERNEL(min_uint8_4x16_neon, u8x16, dummy_t, uint8_t, 16);
MIN_MAX_KERNEL(min_int8_4x16_neon, s8x16, dummy_t, int8_t, 16);

MIN_MAX_KERNEL(max_fp32_4x4_neon, dummy_t, f32x4, float, 4);
MIN_MAX_KERNEL(max_bf16_4x8_neon, dummy_t, bf16x8_rvar, bfloat16, 8);
MIN_MAX_KERNEL(max_fp16_4x8_neon, dummy_t, f16x8_rvar, half, 8);
MIN_MAX_KERNEL(max_uint8_4x16_neon, dummy_t, u8x16, uint8_t, 16);
MIN_MAX_KERNEL(max_int8_4x16_neon, dummy_t, s8x16, int8_t, 16);

void sum_fp32_4x4_neon(size_t n, size_t k3, size_t k2, size_t k1,
                       size_t a_stride_n, size_t a_stride_k3,
                       size_t a_stride_k2, const void* a, size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(float)) {
    tiled_reduce<sum_accumulator_k1_1<f32x4>, float, float>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const float*>(a),
        /*C_stride_m=*/0, reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<accumulator_fp32, float, float>(
      n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
      reinterpret_cast<const float*>(a), /*C_stride_m=*/0,
      reinterpret_cast<float*>(c));
  }
}

}  // namespace ynn
