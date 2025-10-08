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

namespace ynn {

namespace {

using simd::bf16x8;
using simd::f16x8;
using simd::f32x4;
using simd::s16x8;
using simd::s32x4;
using simd::s8x16;
using simd::u8x16;

using f16x8_rvar = float16_wrapper<f16x8, s16x8>;
using bf16x8_rvar = float16_wrapper<bf16x8, s16x8>;

// Use psadbw to compute the absolute difference of a and 0, summing 8 of them
// and producing an int64 in their place. We reinterpret the result to be 4
// int32s, which is only correct because we will do a horizontal total reduction
// later.
YNN_ALWAYS_INLINE s32x4 horizontal_add_8x(u8x16 a) {
  return s32x4{_mm_sad_epu8(a.v, _mm_set1_epi8(0))};
}

// psadbw only exists for unsigned values. We can still use it for signed values
// by toggling the most significant bit, which adds 0x80 to the result. We can
// correct the reduction by subtracting that elsewhere.
YNN_ALWAYS_INLINE s32x4 horizontal_add_8x(s8x16 a) {
  return s32x4{
      _mm_sad_epu8(_mm_xor_si128(a.v, _mm_set1_epi8(0x80)), _mm_set1_epi8(0))};
}

template <bool IsSigned>
struct accumulator_int32 {
  static constexpr std::integral_constant<size_t, 4> N = {};
  static constexpr std::integral_constant<size_t, 16> K = {};

  s32x4 acc[N];

  accumulator_int32() = default;

  YNN_ALWAYS_INLINE explicit accumulator_int32(int32_t k) {
    for (size_t i = 0; i < N; ++i) {
      // We rewrite signed int8 as unsigned in this accumulator. To compensate
      // for this, we need to subtract 0x80 for each element of the reduction.
      // Since this value gets reduced by 4x, we want to subtract 0x20 for each
      // element of the reduction (for a total of 0x80).
      acc[i] = IsSigned ? -(k * 0x20) : 0;
    }
  }

  template <typename AT, typename NT, typename KT>
  YNN_ALWAYS_INLINE void reduce(const AT* A, size_t a_stride_n, NT n, KT k) {
    // This value both identifies what we want the padding to be when we load
    // a partial vector of k values, and indicates the type of the load.
    const simd::vec<AT, K> zero(IsSigned ? 0x80 : 0);
    auto a_0 = load(offset_bytes(A, 0 * a_stride_n), zero, k);
    auto a_1 = 1 < n ? load(offset_bytes(A, 1 * a_stride_n), zero, k) : 0;
    auto a_2 = 2 < n ? load(offset_bytes(A, 2 * a_stride_n), zero, k) : 0;
    auto a_3 = 3 < n ? load(offset_bytes(A, 3 * a_stride_n), zero, k) : 0;
    acc[0] = acc[0] + horizontal_add_8x(a_0);
    acc[1] = acc[1] + horizontal_add_8x(a_1);
    acc[2] = acc[2] + horizontal_add_8x(a_2);
    acc[3] = acc[3] + horizontal_add_8x(a_3);
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
  tiled_reduce<accumulator_int32<true>, int8_t, int32_t>(
      n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
      reinterpret_cast<const int8_t*>(a), /*C_stride_m=*/0,
      reinterpret_cast<int32_t*>(c));
}

void sum_uint8_int32_4x16_sse2(size_t n, size_t k3, size_t k2, size_t k1,
                               size_t a_stride_n, size_t a_stride_k3,
                               size_t a_stride_k2, const void* a, size_t,
                               void* c) {
  tiled_reduce<accumulator_int32<false>, uint8_t, int32_t>(
      n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
      reinterpret_cast<const uint8_t*>(a), /*C_stride_m=*/0,
      reinterpret_cast<int32_t*>(c));
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
