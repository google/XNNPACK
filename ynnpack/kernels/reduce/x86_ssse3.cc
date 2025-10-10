// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/simd/x86_sse.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/reduce.h"

namespace ynn {

namespace {

using simd::s32x4;
using simd::s8x16;
using simd::u8x16;

// Horizontally add 4 values in a, producing an int32.
s32x4 horizontal_add_4x(s8x16 a) {
  __m128i a2x = _mm_maddubs_epi16(_mm_set1_epi8(1), a.v);
  return s32x4{_mm_madd_epi16(_mm_set1_epi16(1), a2x)};
}

s32x4 horizontal_add_4x(u8x16 a) {
  __m128i a2x = _mm_maddubs_epi16(a.v, _mm_set1_epi8(1));
  return s32x4{_mm_madd_epi16(_mm_set1_epi16(1), a2x)};
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
  YNN_ALWAYS_INLINE void reduce(const AT* A, size_t a_stride_n, NT n, KT k) {
    const simd::vec<AT, K> zero(0);
    auto a_0 = load(offset_bytes(A, 0 * a_stride_n), zero, k);
    auto a_1 = 1 < n ? load(offset_bytes(A, 1 * a_stride_n), zero, k) : 0;
    auto a_2 = 2 < n ? load(offset_bytes(A, 2 * a_stride_n), zero, k) : 0;
    auto a_3 = 3 < n ? load(offset_bytes(A, 3 * a_stride_n), zero, k) : 0;
    acc[0] = acc[0] + horizontal_add_4x(a_0);
    acc[1] = acc[1] + horizontal_add_4x(a_1);
    acc[2] = acc[2] + horizontal_add_4x(a_2);
    acc[3] = acc[3] + horizontal_add_4x(a_3);
  }

  template <typename NT>
  YNN_ALWAYS_INLINE void accumulate(size_t /*C_stride_m*/,
                                    int32_t* __restrict C, NT n) {
    std::array<s32x4, 4> acc_t =
        simd::transpose<int32_t>({{acc[0], acc[1], acc[2], acc[3]}});
    s32x4 sum = (acc_t[0] + acc_t[1]) + (acc_t[2] + acc_t[3]);
    store(C, load(C, s32x4{}, n) + sum, n);
  }
};

}  // namespace

void sum_int8_int32_4x16_ssse3(size_t n, size_t k3, size_t k2, size_t k1,
                               size_t a_stride_n, size_t a_stride_k3,
                               size_t a_stride_k2, const void* a, size_t,
                               void* c) {
  tiled_reduce<accumulator_int32, int8_t, int32_t>(
      n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
      reinterpret_cast<const int8_t*>(a), /*C_stride_m=*/0,
      reinterpret_cast<int32_t*>(c));
}

void sum_uint8_int32_4x16_ssse3(size_t n, size_t k3, size_t k2, size_t k1,
                                size_t a_stride_n, size_t a_stride_k3,
                                size_t a_stride_k2, const void* a, size_t,
                                void* c) {
  tiled_reduce<accumulator_int32, uint8_t, int32_t>(
      n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
      reinterpret_cast<const uint8_t*>(a), /*C_stride_m=*/0,
      reinterpret_cast<int32_t*>(c));
}

}  // namespace ynn
