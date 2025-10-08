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
#include "ynnpack/kernels/reduce/reduce.h"

namespace ynn {

namespace {

using simd::s32x4;
using simd::s8x16;
using simd::u8x16;

s32x4 horizontal_add_4x(s32x4 acc, s8x16 a) {
  return s32x4{vdotq_s32(acc.v, a.v, vdupq_n_s8(1))};
}

// We want to accumulate uint8 dot products in int32 accumulators.
s32x4 horizontal_add_4x(s32x4 acc, u8x16 a) {
  return s32x4{vreinterpretq_s32_u32(
      vdotq_u32(vreinterpretq_u32_s32(acc.v), a.v, vdupq_n_u8(1)))};
}

struct accumulator_int32 {
  static constexpr std::integral_constant<size_t, 4> N = {};
  static constexpr std::integral_constant<size_t, 16> K = {};

  s32x4 acc[N];

  accumulator_int32() = default;

  YNN_ALWAYS_INLINE explicit accumulator_int32(size_t) {
    for (size_t i = 0; i < N; ++i) {
      acc[i] = 0;
    }
  }

  template <typename AT, typename NT, typename KT>
  YNN_ALWAYS_INLINE void reduce(const AT* A, size_t A_stride_n, NT n, KT k) {
    const simd::vec<AT, K> zero(0);
    auto a_0 = load(offset_bytes(A, 0 * A_stride_n), zero, k);
    auto a_1 = 1 < n ? load(offset_bytes(A, 1 * A_stride_n), zero, k) : 0;
    auto a_2 = 2 < n ? load(offset_bytes(A, 2 * A_stride_n), zero, k) : 0;
    auto a_3 = 3 < n ? load(offset_bytes(A, 3 * A_stride_n), zero, k) : 0;
    acc[0] = horizontal_add_4x(acc[0], a_0);
    acc[1] = horizontal_add_4x(acc[1], a_1);
    acc[2] = horizontal_add_4x(acc[2], a_2);
    acc[3] = horizontal_add_4x(acc[3], a_3);
  }

  template <typename NT>
  YNN_ALWAYS_INLINE void accumulate(size_t /*C_stride_m*/,
                                    int32_t* __restrict C, NT n) {
    // We have 4 accumulators that each need to be reduced. It's probably best
    // to do this by transposing and then doing a SIMD reduction, instead of
    // doing in-vector reductions.
    std::array<s32x4, 4> acc_t =
        simd::transpose<int32_t>({{acc[0], acc[1], acc[2], acc[3]}});
    s32x4 sum = (acc_t[0] + acc_t[1]) + (acc_t[2] + acc_t[3]);
    store(C, load(C, s32x4{}, n) + sum, n);
  }
};

}  // namespace

void sum_int8_int32_4x16_neondot(size_t n, size_t k3, size_t k2, size_t k1,
                                 size_t a_stride_n, size_t a_stride_k3,
                                 size_t a_stride_k2, const void* a, size_t,
                                 void* c) {
  tiled_reduce<accumulator_int32, int8_t, int32_t>(
      n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
      reinterpret_cast<const int8_t*>(a), /*C_stride_m=*/0,
      reinterpret_cast<int32_t*>(c));
}

void sum_uint8_int32_4x16_neondot(size_t n, size_t k3, size_t k2, size_t k1,
                                  size_t a_stride_n, size_t a_stride_k3,
                                  size_t a_stride_k2, const void* a, size_t,
                                  void* c) {
  tiled_reduce<accumulator_int32, uint8_t, int32_t>(
      n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
      reinterpret_cast<const uint8_t*>(a), /*C_stride_m=*/0,
      reinterpret_cast<int32_t*>(c));
}

}  // namespace ynn
