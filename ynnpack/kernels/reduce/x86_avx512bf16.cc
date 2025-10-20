// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <cstring>
#include <type_traits>

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/simd/multi_vec.h"
#include "ynnpack/base/simd/x86_avx.h"
#include "ynnpack/base/simd/x86_avx512.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace simd {

using f32x16x2 = multi_vec<f32x16, 2>;

static f32x16& operator+=(f32x16& a, bf16x16 b) {
  a.v = _mm512_add_ps(a.v, _mm512_cvtpbh_ps(reinterpret_cast<__m256bh>(b.v)));
  return a;
}

static f32x16x2& operator+=(f32x16x2& a, bf16x32 b) {
  a.v[0] += extract<0>(b, bf16x16{});
  a.v[1] += extract<1>(b, bf16x16{});
  return a;
}

static f32x16& reduce_add(
    f32x16& a, bf16x32 b,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  f32x16 b_f32(_mm512_dpbf16_ps(
    _mm512_setzero_ps(), reinterpret_cast<__m512bh>(b.v),
    reinterpret_cast<__m512bh>(_mm512_set1_epi16(bfloat16(1.0f).to_bits()))));
  return a += b_f32;
}

}  // namespace simd

using simd::f32x16x2;
using simd::f32x16;
using simd::bf16x32;

void sum_bf16_fp32_4x32_avx512bf16(size_t n, size_t k3, size_t k2, size_t k1,
                                   size_t a_stride_n, size_t a_stride_k3,
                                   size_t a_stride_k2, const void* a, size_t,
                                   void* c) {
  if (k1 == 1 && a_stride_n == sizeof(bfloat16)) {
    tiled_reduce<sum_accumulator_k1_1<bf16x32, f32x16x2>, bfloat16, float>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a),
        /*C_stride_m=*/0, reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<f32x16, 32>, bfloat16, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

}  // namespace ynn
