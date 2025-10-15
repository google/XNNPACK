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
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/x86_sse.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max_accumulator.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"
#include "ynnpack/kernels/reduce/x86_sse.h"

namespace ynn {

using simd::s32x16;

namespace simd {

YNN_ALWAYS_INLINE s32x16& operator+=(s32x16& a, s8x16 b) {
  __m128i i8_lo = _mm_unpacklo_epi8(b.v, b.v);
  __m128i i8_hi = _mm_unpackhi_epi8(b.v, b.v);

  s32x4 b_0(_mm_srai_epi32(_mm_unpacklo_epi16(i8_lo, i8_lo), 24));
  s32x4 b_1(_mm_srai_epi32(_mm_unpackhi_epi16(i8_lo, i8_lo), 24));
  s32x4 b_2(_mm_srai_epi32(_mm_unpacklo_epi16(i8_hi, i8_hi), 24));
  s32x4 b_3(_mm_srai_epi32(_mm_unpackhi_epi16(i8_hi, i8_hi), 24));

  a.v[0].v[0] += b_0;
  a.v[0].v[1] += b_1;
  a.v[1].v[0] += b_2;
  a.v[1].v[1] += b_3;
  return a;
}

YNN_ALWAYS_INLINE s32x16& operator+=(s32x16& a, u8x16 b) {
  const __m128i zero = _mm_setzero_si128();
  __m128i i16_lo = _mm_unpacklo_epi8(b.v, zero);
  __m128i i16_hi = _mm_unpackhi_epi8(b.v, zero);

  s32x4 b_0(_mm_unpacklo_epi16(i16_lo, zero));
  s32x4 b_1(_mm_unpackhi_epi16(i16_lo, zero));
  s32x4 b_2(_mm_unpacklo_epi16(i16_hi, zero));
  s32x4 b_3(_mm_unpackhi_epi16(i16_hi, zero));

  a.v[0].v[0] += b_0;
  a.v[0].v[1] += b_1;
  a.v[1].v[0] += b_2;
  a.v[1].v[1] += b_3;
  return a;
}

}  // namespace simd

namespace {

using simd::bf16x8;
using simd::f16x8;
using simd::f32x4;
using simd::s16x8;

using f16x8_rvar = float16_wrapper<f16x8, s16x8>;
using bf16x8_rvar = float16_wrapper<bf16x8, s16x8>;

struct accumulator_fp32 {
  static constexpr std::integral_constant<size_t, 4> N = {};
  static constexpr std::integral_constant<size_t, 4> K = {};

  f32x4 acc[N];

  accumulator_fp32() = default;

  YNN_ALWAYS_INLINE explicit accumulator_fp32(int32_t k) {
    for (size_t i = 0; i < N; ++i) {
      acc[i] = 0;
    }
  }

  template <typename NT, typename KT>
  YNN_ALWAYS_INLINE void reduce(const float* A, size_t a_stride_n, NT n, KT k) {
    const simd::vec<float, K> zero(0);
    auto a_0 = load(offset_bytes(A, 0 * a_stride_n), zero, k);
    auto a_1 = 1 < n ? load(offset_bytes(A, 1 * a_stride_n), zero, k) : 0;
    auto a_2 = 2 < n ? load(offset_bytes(A, 2 * a_stride_n), zero, k) : 0;
    auto a_3 = 3 < n ? load(offset_bytes(A, 3 * a_stride_n), zero, k) : 0;
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

MIN_MAX_KERNEL(min_max_fp32_4x4_sse2, f32x4, f32x4, float, 4);
MIN_MAX_KERNEL(min_max_bf16_4x8_sse2, bf16x8_rvar, bf16x8_rvar, bfloat16, 8);
MIN_MAX_KERNEL(min_max_fp16_4x8_sse2, f16x8_rvar, f16x8_rvar, half, 8);
MIN_MAX_KERNEL(min_max_uint8_4x16_sse2, u8x16, u8x16, uint8_t, 16);

MIN_MAX_KERNEL(min_fp32_4x4_sse2, f32x4, dummy_t, float, 4);
MIN_MAX_KERNEL(min_bf16_4x8_sse2, bf16x8_rvar, dummy_t, bfloat16, 8);
MIN_MAX_KERNEL(min_fp16_4x8_sse2, f16x8_rvar, dummy_t, half, 8);
MIN_MAX_KERNEL(min_uint8_4x16_sse2, u8x16, dummy_t, uint8_t, 16);

MIN_MAX_KERNEL(max_fp32_4x4_sse2, dummy_t, f32x4, float, 4);
MIN_MAX_KERNEL(max_bf16_4x8_sse2, dummy_t, bf16x8_rvar, bfloat16, 8);
MIN_MAX_KERNEL(max_fp16_4x8_sse2, dummy_t, f16x8_rvar, half, 8);
MIN_MAX_KERNEL(max_uint8_4x16_sse2, dummy_t, u8x16, uint8_t, 16);

void sum_int8_int32_4x16_sse2(size_t n, size_t k3, size_t k2, size_t k1,
                              size_t a_stride_n, size_t a_stride_k3,
                              size_t a_stride_k2, const void* a, size_t,
                              void* c) {
  if (k1 == 1 && a_stride_n == sizeof(int8_t)) {
    tiled_reduce<sum_accumulator_k1_1<s32x16>, int8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const int8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<accumulator_int32<true>, int8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const int8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

void sum_uint8_int32_4x16_sse2(size_t n, size_t k3, size_t k2, size_t k1,
                               size_t a_stride_n, size_t a_stride_k3,
                               size_t a_stride_k2, const void* a, size_t,
                               void* c) {
  if (k1 == 1 && a_stride_n == sizeof(uint8_t)) {
    tiled_reduce<sum_accumulator_k1_1<s32x16>, uint8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<accumulator_int32<false>, uint8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

void sum_fp32_4x4_sse2(size_t n, size_t k3, size_t k2, size_t k1,
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
