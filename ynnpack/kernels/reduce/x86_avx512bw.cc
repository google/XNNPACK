// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/simd/x86_avx.h"
#include "ynnpack/base/simd/x86_avx512.h"
#include "ynnpack/base/simd/x86_sse.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max_accumulator.h"
#include "ynnpack/kernels/reduce/reduce.h"

namespace ynn {

namespace {

using simd::bf16x32;
using simd::extract;
using simd::f16x32;
using simd::s16x32;
using simd::s32x16;
using simd::s32x4;
using simd::s32x8;
using simd::s8x64;
using simd::u8x64;

using f16x32_rvar = float16_wrapper<f16x32, s16x32>;
using bf16x32_rvar = float16_wrapper<bf16x32, s16x32>;

s32x16 horizontal_add_4x(s8x64 a) {
  __m512i a2x = _mm512_maddubs_epi16(_mm512_set1_epi8(1), a.v);
  return s32x16{_mm512_madd_epi16(_mm512_set1_epi16(1), a2x)};
}

s32x16 horizontal_add_4x(u8x64 a) {
  __m512i a2x = _mm512_maddubs_epi16(a.v, _mm512_set1_epi8(1));
  return s32x16{_mm512_madd_epi16(_mm512_set1_epi16(1), a2x)};
}

s32x8 horizontal_add_2x(s32x8 a, s32x8 b) {
  return s32x8{_mm256_hadd_epi32(a.v, b.v)};
}

s32x16 horizontal_add_2x(s32x16 a, s32x16 b) {
  static const __m512i perm_idx0 =
      _mm512_setr_epi32(0x00, 0x02, 0x04, 0x06, 0x08, 0x0A, 0x0C, 0x0E, 0x10,
                        0x12, 0x14, 0x16, 0x18, 0x1A, 0x1C, 0x1E);
  static const __m512i perm_idx1 =
      _mm512_setr_epi32(0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F, 0x11,
                        0x13, 0x15, 0x17, 0x19, 0x1B, 0x1D, 0x1F);
  __m512i ab0 = _mm512_permutex2var_epi32(a.v, perm_idx0, b.v);
  __m512i ab1 = _mm512_permutex2var_epi32(a.v, perm_idx1, b.v);
  return s32x16{_mm512_add_epi32(ab0, ab1)};
}

struct accumulator_int32 {
  static constexpr std::integral_constant<size_t, 4> N = {};
  static constexpr std::integral_constant<size_t, 64> K = {};

  s32x16 acc[N];

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
    auto a_1 = 1 < n ? load(offset_bytes(A, 1 * a_stride_n), zero, k) : zero;
    auto a_2 = 2 < n ? load(offset_bytes(A, 2 * a_stride_n), zero, k) : zero;
    auto a_3 = 3 < n ? load(offset_bytes(A, 3 * a_stride_n), zero, k) : zero;
    acc[0] = acc[0] + horizontal_add_4x(a_0);
    acc[1] = acc[1] + horizontal_add_4x(a_1);
    acc[2] = acc[2] + horizontal_add_4x(a_2);
    acc[3] = acc[3] + horizontal_add_4x(a_3);
  }

  template <typename NT>
  YNN_ALWAYS_INLINE void accumulate(size_t /*C_stride_m*/,
                                    int32_t* __restrict C, NT n) {
    const s32x16 sum01 = horizontal_add_2x(acc[0], acc[1]);
    const s32x16 sum23 = horizontal_add_2x(acc[2], acc[3]);
    const s32x16 sum0123 = horizontal_add_2x(sum01, sum23);
    const s32x16 suml1 = horizontal_add_2x(sum0123, sum0123);
    const s32x8 suml1_lo = extract<0>(suml1, s32x8{});
    const s32x8 suml2 = horizontal_add_2x(suml1_lo, suml1_lo);
    const s32x4 sum{_mm_unpacklo_epi64(extract<0>(suml2, s32x4{}).v,
                                       extract<1>(suml2, s32x4{}).v)};
    store(C, load(C, s32x4{}, n) + sum, n);
  }
};

}  // namespace

MIN_MAX_KERNEL(min_max_bf16_4x32_avx512bw, bf16x32_rvar, bf16x32_rvar, bfloat16,
               32);
MIN_MAX_KERNEL(min_max_fp16_4x32_avx512bw, f16x32_rvar, f16x32_rvar, half, 32);
MIN_MAX_KERNEL(min_max_uint8_4x64_avx512bw, u8x64, u8x64, uint8_t, 64);
MIN_MAX_KERNEL(min_max_int8_4x64_avx512bw, s8x64, s8x64, int8_t, 64);

MIN_MAX_KERNEL(min_bf16_4x32_avx512bw, bf16x32_rvar, dummy_t, bfloat16, 32);
MIN_MAX_KERNEL(min_fp16_4x32_avx512bw, f16x32_rvar, dummy_t, half, 32);
MIN_MAX_KERNEL(min_uint8_4x64_avx512bw, u8x64, dummy_t, uint8_t, 64);
MIN_MAX_KERNEL(min_int8_4x64_avx512bw, s8x64, dummy_t, int8_t, 64);

MIN_MAX_KERNEL(max_bf16_4x32_avx512bw, dummy_t, bf16x32_rvar, bfloat16, 32);
MIN_MAX_KERNEL(max_fp16_4x32_avx512bw, dummy_t, f16x32_rvar, half, 32);
MIN_MAX_KERNEL(max_uint8_4x64_avx512bw, dummy_t, u8x64, uint8_t, 64);
MIN_MAX_KERNEL(max_int8_4x64_avx512bw, dummy_t, s8x64, int8_t, 64);

void sum_int8_int32_4x64_avx512bw(size_t n, size_t k3, size_t k2, size_t k1,
                                  size_t a_stride_n, size_t a_stride_k3,
                                  size_t a_stride_k2, const void* a, size_t,
                                  void* c) {
  tiled_reduce<accumulator_int32, int8_t, int32_t>(
      n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
      reinterpret_cast<const int8_t*>(a), /*C_stride_m=*/0,
      reinterpret_cast<int32_t*>(c));
}

void sum_uint8_int32_4x64_avx512bw(size_t n, size_t k3, size_t k2, size_t k1,
                                   size_t a_stride_n, size_t a_stride_k3,
                                   size_t a_stride_k2, const void* a, size_t,
                                   void* c) {
  tiled_reduce<accumulator_int32, uint8_t, int32_t>(
      n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
      reinterpret_cast<const uint8_t*>(a), /*C_stride_m=*/0,
      reinterpret_cast<int32_t*>(c));
}

}  // namespace ynn
