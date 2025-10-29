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
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace simd {

using f32x8x2 = multi_vec<f32x8, 2>;
using f32x8x16 = multi_vec<f32x8, 16>;
using f16x16x8 = multi_vec<f16x16, 8>;

static f32x8& operator+=(f32x8& a, f16x8 b) {
  a.v = _mm256_add_ps(a.v, _mm256_cvtph_ps(b.v));
  return a;
}

static f32x8x2& operator+=(f32x8x2& a, f16x16 b) {
    a.v[0] += extract<0>(b, f16x8{});
    a.v[1] += extract<1>(b, f16x8{});
    return a;
}

static f32x8x16& operator+=(f32x8x16& a, f16x16x8 b) {
  YNN_UNROLL
  for (size_t i = 0; i < 8; ++i) {
    a.v[2 * i] += extract<0>(b.v[i], f16x8{});
    a.v[2 * i + 1] += extract<1>(b.v[i], f16x8{});
  }
  return a;
}

template <int Index>
static f32x4 extract(f32x8x2 x, f32x4) {
  return extract<Index % 2>(x.v[Index / 2], f32x4{});
}

}  // namespace simd

using simd::f32x8x2;
using simd::f32x8x16;
using simd::f16x16x8;

void sum_fp16_fp32_f16c(size_t n, size_t k3, size_t k2, size_t k1,
                        size_t a_stride_n, size_t a_stride_k3,
                        size_t a_stride_k2, const void* a, size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(half)) {
    tiled_reduce<sum_accumulator_k1_1<f16x16x8, f32x8x16>, half, float>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const half*>(a),
        /*C_stride_m=*/0, reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<f32x8x2, 16>, half, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const half*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

}  // namespace ynn
