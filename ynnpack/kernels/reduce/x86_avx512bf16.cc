// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <cstddef>
#include <type_traits>

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/simd/x86_avx512.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/sum.h"

namespace ynn {

namespace simd {

static f32x32 reduce_add(
    f32x32 a, bf16x32 b, identity /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  __m512bh ones = reinterpret_cast<__m512bh>(_mm512_set1_epi32(0x00003F80));
  __m512i b_bits = reinterpret_cast<__m512i>(b.v);
  __m512bh lo = reinterpret_cast<__m512bh>(
      _mm512_cvtepu16_epi32(_mm512_castsi512_si256(b_bits)));
  __m512bh hi = reinterpret_cast<__m512bh>(
      _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(b_bits, 1)));

  return concat(f32x16{_mm512_dpbf16_ps(extract<0>(a, f32x16::N).v, lo, ones)},
                f32x16{_mm512_dpbf16_ps(extract<1>(a, f32x16::N).v, hi, ones)});
}

static f32x32 reduce_add(
    f32x32 a, bf16x32 b, square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  __m512i b_bits = reinterpret_cast<__m512i>(b.v);
  __m512bh lo = reinterpret_cast<__m512bh>(
      _mm512_cvtepu16_epi32(_mm512_castsi512_si256(b_bits)));
  __m512bh hi = reinterpret_cast<__m512bh>(
      _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(b_bits, 1)));

  return concat(f32x16{_mm512_dpbf16_ps(extract<0>(a, f32x16::N).v, lo, lo)},
                f32x16{_mm512_dpbf16_ps(extract<1>(a, f32x16::N).v, hi, hi)});
}

static f32x16 reduce_add(
    f32x16 a, bf16x32 b, identity /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  return f32x16{_mm512_dpbf16_ps(
      a.v, reinterpret_cast<__m512bh>(b.v),
      reinterpret_cast<__m512bh>(_mm512_set1_epi16(bfloat16(1.0f).to_bits())))};
}

static f32x16 reduce_add(
    f32x16 a, bf16x32 b, square /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  return f32x16{_mm512_dpbf16_ps(
      a.v, reinterpret_cast<__m512bh>(b.v), reinterpret_cast<__m512bh>(b.v))};
}

}  // namespace simd

using simd::bf16x32;
using simd::f32x16;
using simd::f32x32;

SUM_FLOAT_K1_KERNEL(sum_k1_bf16_fp32_avx512bf16, bfloat16, float, 0, 2,
                    identity);
SUM_FLOAT_KN_KERNEL(sum_kn_bf16_fp32_avx512bf16, bfloat16, float, 32, identity);

SUM_FLOAT_K1_KERNEL(sum_squared_k1_bf16_fp32_avx512bf16, bfloat16, float, 0, 2,
                    square);
SUM_FLOAT_KN_KERNEL(sum_squared_kn_bf16_fp32_avx512bf16, bfloat16, float, 32,
                    square);

}  // namespace ynn
