// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <cstring>
#include <type_traits>

#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/simd/multi_vec.h"
#include "ynnpack/base/simd/x86_avx2.h"
#include "ynnpack/base/simd/x86_avx512bw.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace simd {

using f32x16x8 = multi_vec<f32x16, 8>;
using bf16x32x4 = multi_vec<bf16x32, 4>;

static f32x16& operator+=(f32x16& a, bf16x16 b) {
  return a += f32x16{_mm512_cvtpbh_ps(reinterpret_cast<__m256bh>(b.v))};
}

static f32x16x8& operator+=(f32x16x8& a, bf16x32x4 b) {
  YNN_UNROLL
  for (size_t i = 0; i < 4; ++i) {
    a.v[2 * i] += extract<0>(b.v[i], bf16x16{});
    a.v[2 * i + 1] += extract<1>(b.v[i], bf16x16{});
  }
  return a;
}

static f32x16 reduce_add(
    f32x16 a, bf16x32 b, Identity /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  return f32x16{_mm512_dpbf16_ps(
      a.v, reinterpret_cast<__m512bh>(b.v),
      reinterpret_cast<__m512bh>(_mm512_set1_epi16(bfloat16(1.0f).to_bits())))};
}

static f32x16x8 reduce_add(
    f32x16x8 a, bf16x32x4 b, Square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  f32x16x8 result = a;

  YNN_UNROLL
  for (int i = 0; i < 4; ++i) {
    __m512i b_bits = reinterpret_cast<__m512i>(b.v[i].v);

    __m256i lower_half = _mm512_castsi512_si256(b_bits);
    __m256i upper_half = _mm512_extracti64x4_epi64(b_bits, 1);

    __m512i expanded_lo_i = _mm512_cvtepu16_epi32(lower_half);
    __m512i expanded_hi_i = _mm512_cvtepu16_epi32(upper_half);

    __m512bh expanded_lo = reinterpret_cast<__m512bh>(expanded_lo_i);
    __m512bh expanded_hi = reinterpret_cast<__m512bh>(expanded_hi_i);

    result.v[2 * i].v = _mm512_dpbf16_ps(result.v[2 * i].v, expanded_lo,
                                       expanded_lo);
    result.v[2 * i + 1].v = _mm512_dpbf16_ps(result.v[2 * i + 1].v, expanded_hi,
                                             expanded_hi);
  }

  return result;
}

static f32x16 reduce_add(
    f32x16 a, bf16x32 b, Square /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  return f32x16{_mm512_dpbf16_ps(
      a.v, reinterpret_cast<__m512bh>(b.v), reinterpret_cast<__m512bh>(b.v))};
}

}  // namespace simd

using simd::f32x16;
using simd::f32x16x8;
using simd::bf16x32x4;

void sum_bf16_fp32_avx512bf16(size_t n, size_t k3, size_t k2, size_t k1,
                              size_t a_stride_n, size_t a_stride_k3,
                              size_t a_stride_k2, const void* a, size_t,
                              void* c) {
  if (k1 == 1 && a_stride_n == sizeof(bfloat16)) {
    tiled_reduce<sum_accumulator_k1_1<bf16x32x4, f32x16x8>, bfloat16, float>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<f32x16, 32>, bfloat16, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

void sum_squared_bf16_fp32_avx512bf16(size_t n, size_t k3, size_t k2, size_t k1,
                                      size_t a_stride_n, size_t a_stride_k3,
                                      size_t a_stride_k2, const void* a, size_t,
                                      void* c) {
  if (k1 == 1 && a_stride_n == sizeof(bfloat16)) {
    tiled_reduce<sum_accumulator_k1_1<bf16x32x4, f32x16x8, Square>, bfloat16,
      float>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<f32x16, 32, Square>, bfloat16, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

}  // namespace ynn
