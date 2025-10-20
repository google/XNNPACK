// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <cstring>

#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/multi_vec.h"
#include "ynnpack/base/simd/x86_avx.h"
#include "ynnpack/base/simd/x86_avx512.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace simd {

using f32x16x2 = multi_vec<f32x16, 2>;

static f32x16& operator+=(f32x16& a, f16x16 b) {
  a.v = _mm512_add_ps(a.v, _mm512_cvtph_ps(b.v));
  return a;
}

static f32x16x2& operator+=(f32x16x2& a, f16x32 b) {
  a.v[0] += extract<0>(b, f16x16{});
  a.v[1] += extract<1>(b, f16x16{});
  return a;
}

}  // namespace simd

using simd::f16x32;
using simd::f32x16;
using simd::f32x16x2;

void sum_fp16_fp32_4x32_avx512fp16(size_t n, size_t k3, size_t k2, size_t k1,
                                   size_t a_stride_n, size_t a_stride_k3,
                                   size_t a_stride_k2, const void* a, size_t,
                                   void* c) {
  if (k1 == 1 && a_stride_n == sizeof(half)) {
    tiled_reduce<sum_accumulator_k1_1<f16x32, f32x16x2>, half, float>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const half*>(a),
        /*C_stride_m=*/0, reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<f32x16, 16>, half, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const half*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

}  // namespace ynn
