// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/simd/x86_avx.h"
#include "ynnpack/base/simd/x86_avx512.h"
#include "ynnpack/base/simd/x86_sse.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

using simd::f32x16;
using simd::bf16x16;
using simd::bf16x32;
using simd::extract;

YNN_ALWAYS_INLINE f32x16& operator+=(f32x16& a, bf16x16 b) {
  a.v = _mm512_add_ps(a.v, _mm512_cvtpbh_ps(reinterpret_cast<__m256bh>(b.v)));
  return a;
}

namespace {

using simd::f32x4;

f32x16 horizontal_add_2x(bf16x32 a) {
  return f32x16{_mm512_dpbf16_ps(
      _mm512_setzero_ps(), reinterpret_cast<__m512bh>(a.v),
      reinterpret_cast<__m512bh>(_mm512_set1_epi16(bfloat16(1.0f).to_bits())))};
}

struct accumulator_fp32 {
  static constexpr std::integral_constant<size_t, 4> N = {};
  static constexpr std::integral_constant<size_t, 32> K = {};

  f32x16 acc[N];

  accumulator_fp32() = default;

  YNN_ALWAYS_INLINE explicit accumulator_fp32(size_t k) {
    for (size_t i = 0; i < N; ++i) {
      acc[i] = 0;
    }
  }

  template <typename NT, typename KT>
  YNN_ALWAYS_INLINE void reduce(const bfloat16* A, size_t a_stride_n, NT n,
                                KT k) {
    const simd::vec<bfloat16, K> zero(bfloat16(0.0f));
    auto a_0 = load(offset_bytes(A, 0 * a_stride_n), zero, k);
    auto a_1 = 1 < n ? load(offset_bytes(A, 1 * a_stride_n), zero, k) : zero;
    auto a_2 = 2 < n ? load(offset_bytes(A, 2 * a_stride_n), zero, k) : zero;
    auto a_3 = 3 < n ? load(offset_bytes(A, 3 * a_stride_n), zero, k) : zero;
    acc[0] += horizontal_add_2x(a_0);
    acc[1] += horizontal_add_2x(a_1);
    acc[2] += horizontal_add_2x(a_2);
    acc[3] += horizontal_add_2x(a_3);
  }

  template <typename NT>
  YNN_ALWAYS_INLINE void accumulate(size_t /*C_stride_m*/, float* __restrict C,
                                    NT n) {
    auto v_0 = (extract<0>(acc[0], f32x4{}) + extract<1>(acc[0], f32x4{})) +
               (extract<2>(acc[0], f32x4{}) + extract<3>(acc[0], f32x4{}));
    auto v_1 = (extract<0>(acc[1], f32x4{}) + extract<1>(acc[1], f32x4{})) +
               (extract<2>(acc[1], f32x4{}) + extract<3>(acc[1], f32x4{}));
    auto v_2 = (extract<0>(acc[2], f32x4{}) + extract<1>(acc[2], f32x4{})) +
               (extract<2>(acc[2], f32x4{}) + extract<3>(acc[2], f32x4{}));
    auto v_3 = (extract<0>(acc[3], f32x4{}) + extract<1>(acc[3], f32x4{})) +
               (extract<2>(acc[3], f32x4{}) + extract<3>(acc[3], f32x4{}));

    auto t = simd::transpose<float>({{v_0, v_1, v_2, v_3}});
    const f32x4 sum = (t[0] + t[1]) + (t[2] + t[3]);
    store(C, load(C, f32x4{}, n) + sum, n);
  }
};

}  // namespace

void sum_bf16_fp32_4x32_avx512bf16(size_t n, size_t k3, size_t k2, size_t k1,
                                   size_t a_stride_n, size_t a_stride_k3,
                                   size_t a_stride_k2, const void* a, size_t,
                                   void* c) {
  if (k1 == 1 && a_stride_n == sizeof(bfloat16)) {
    tiled_reduce<sum_accumulator_k1_1<f32x16>, bfloat16, float>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a),
        /*C_stride_m=*/0, reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<accumulator_fp32, bfloat16, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

}  // namespace ynn
