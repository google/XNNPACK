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
#include "ynnpack/base/simd/x86_avx.h"
#include "ynnpack/base/simd/x86_sse.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max_accumulator.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace {

using simd::bf16x16;
using simd::extract;
using simd::f16x16;
using simd::f32x4;
using simd::f32x8;
using simd::s16x16;
using simd::s32x4;
using simd::s32x8;
using simd::s8x32;
using simd::u8x32;

using f16x16_rvar = float16_wrapper<f16x16, s16x16>;
using bf16x16_rvar = float16_wrapper<bf16x16, s16x16>;

s32x8 horizontal_add_4x(s8x32 a) {
  __m256i a2x = _mm256_maddubs_epi16(_mm256_set1_epi8(1), a.v);
  return s32x8{_mm256_madd_epi16(_mm256_set1_epi16(1), a2x)};
}

s32x8 horizontal_add_4x(u8x32 a) {
  __m256i a2x = _mm256_maddubs_epi16(a.v, _mm256_set1_epi8(1));
  return s32x8{_mm256_madd_epi16(_mm256_set1_epi16(1), a2x)};
}

struct accumulator_int32 {
  static constexpr std::integral_constant<size_t, 4> N = {};
  static constexpr std::integral_constant<size_t, 32> K = {};

  s32x8 acc[N];

  accumulator_int32() = default;

  YNN_ALWAYS_INLINE explicit accumulator_int32(size_t k) {
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
    auto low = simd::transpose<int32_t>({{
        extract<0>(acc[0], s32x4{}),
        extract<0>(acc[1], s32x4{}),
        extract<0>(acc[2], s32x4{}),
        extract<0>(acc[3], s32x4{}),
    }});
    auto high = simd::transpose<int32_t>({{
        extract<1>(acc[0], s32x4{}),
        extract<1>(acc[1], s32x4{}),
        extract<1>(acc[2], s32x4{}),
        extract<1>(acc[3], s32x4{}),
    }});
    const s32x4 sum = ((low[0] + high[0]) + (low[1] + high[1])) +
                      ((low[2] + high[2]) + (low[3] + high[3]));
    store(C, load(C, s32x4{}, n) + sum, n);
  }
};

struct accumulator_fp32 {
  static constexpr std::integral_constant<size_t, 4> N = {};
  static constexpr std::integral_constant<size_t, 8> K = {};

  f32x8 acc[N];

  accumulator_fp32() = default;

  YNN_ALWAYS_INLINE explicit accumulator_fp32(size_t k) {
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
    auto low = simd::transpose<float>({{
        extract<0>(acc[0], f32x4{}),
        extract<0>(acc[1], f32x4{}),
        extract<0>(acc[2], f32x4{}),
        extract<0>(acc[3], f32x4{}),
    }});
    auto high = simd::transpose<float>({{
        extract<1>(acc[0], f32x4{}),
        extract<1>(acc[1], f32x4{}),
        extract<1>(acc[2], f32x4{}),
        extract<1>(acc[3], f32x4{}),
    }});
    const f32x4 sum = ((low[0] + high[0]) + (low[1] + high[1])) +
                      ((low[2] + high[2]) + (low[3] + high[3]));
    store(C, load(C, f32x4{}, n) + sum, n);
  }
};

}  // namespace

MIN_MAX_KERNEL(min_max_fp32_4x8_avx2, f32x8, f32x8, float, 8);
MIN_MAX_KERNEL(min_max_bf16_4x16_avx2, bf16x16_rvar, bf16x16_rvar, bfloat16,
               16);
MIN_MAX_KERNEL(min_max_fp16_4x16_avx2, f16x16_rvar, f16x16_rvar, half, 16);
MIN_MAX_KERNEL(min_max_uint8_4x32_avx2, u8x32, u8x32, uint8_t, 32);
MIN_MAX_KERNEL(min_max_int8_4x32_avx2, s8x32, s8x32, int8_t, 32);

MIN_MAX_KERNEL(min_fp32_4x8_avx2, f32x8, dummy_t, float, 8);
MIN_MAX_KERNEL(min_bf16_4x16_avx2, bf16x16_rvar, dummy_t, bfloat16, 16);
MIN_MAX_KERNEL(min_fp16_4x16_avx2, f16x16_rvar, dummy_t, half, 16);
MIN_MAX_KERNEL(min_uint8_4x32_avx2, u8x32, dummy_t, uint8_t, 32);
MIN_MAX_KERNEL(min_int8_4x32_avx2, s8x32, dummy_t, int8_t, 32);

MIN_MAX_KERNEL(max_fp32_4x8_avx2, dummy_t, f32x8, float, 8);
MIN_MAX_KERNEL(max_bf16_4x16_avx2, dummy_t, bf16x16_rvar, bfloat16, 16);
MIN_MAX_KERNEL(max_fp16_4x16_avx2, dummy_t, f16x16_rvar, half, 16);
MIN_MAX_KERNEL(max_uint8_4x32_avx2, dummy_t, u8x32, uint8_t, 32);
MIN_MAX_KERNEL(max_int8_4x32_avx2, dummy_t, s8x32, int8_t, 32);

void sum_int8_int32_4x32_avx2(size_t n, size_t k3, size_t k2, size_t k1,
                              size_t a_stride_n, size_t a_stride_k3,
                              size_t a_stride_k2, const void* a, size_t,
                              void* c) {
  tiled_reduce<accumulator_int32, int8_t, int32_t>(
      n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
      reinterpret_cast<const int8_t*>(a), /*C_stride_m=*/0,
      reinterpret_cast<int32_t*>(c));
}

void sum_uint8_int32_4x32_avx2(size_t n, size_t k3, size_t k2, size_t k1,
                               size_t a_stride_n, size_t a_stride_k3,
                               size_t a_stride_k2, const void* a, size_t,
                               void* c) {
  tiled_reduce<accumulator_int32, uint8_t, int32_t>(
      n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
      reinterpret_cast<const uint8_t*>(a), /*C_stride_m=*/0,
      reinterpret_cast<int32_t*>(c));
}

void sum_fp32_4x8_avx2(size_t n, size_t k3, size_t k2, size_t k1,
                       size_t a_stride_n, size_t a_stride_k3,
                       size_t a_stride_k2, const void* a, size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(float)) {
    tiled_reduce<sum_accumulator_k1_1<f32x8>, float, float>(
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
