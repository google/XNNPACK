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
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/x86_avx.h"
#include "ynnpack/base/simd/x86_sse.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace {

using simd::extract;
using simd::f16x16;
using simd::f16x8;
using simd::f32x4;
using simd::f32x8;

YNN_ALWAYS_INLINE f32x8& operator+=(f32x8& a, f16x8 b) {
  a.v = _mm256_add_ps(a.v, _mm256_cvtph_ps(b.v));
  return a;
}

struct f32x16 {
  f32x8 v[2];

  using value_type = float;
  static constexpr std::integral_constant<size_t, 16> N = {};  // NOLINT

  f32x16() = default;
  explicit f32x16(float x) : v{x, x} {};

  YNN_ALWAYS_INLINE f32x16 operator+(f32x16 a) const {
    f32x16 res;
    res.v[0] = this->v[0] + a.v[0];
    res.v[1] = this->v[1] + a.v[1];
    return res;
  }

  f32x16& operator+=(f16x16 x) {
    v[0] += extract<0>(x, f16x8{});
    v[1] += extract<1>(x, f16x8{});
    return *this;
  }
};

template <int Index>
YNN_ALWAYS_INLINE f32x4 extract(f32x16 x, f32x4) {
  if (Index < 2) {
    return extract<Index>(x.v[0], f32x4{});
  } else {
    return extract<Index % 2>(x.v[1], f32x4{});
  }
}

template <typename NT>
YNN_ALWAYS_INLINE f32x16 load(const float* ptr, f32x16, NT n) {
  f32x16 x;

  if (n < f32x8::N) {
    x.v[0] = load(ptr, f32x8{}, n);
    x.v[1] = f32x8(0.0f);
  } else {
    x.v[0] = load(ptr, f32x8{}, f32x8::N);
    x.v[1] = load(ptr + f32x8::N, f32x8(0.0f), n - f32x8::N);
  }

  return x;
}

template <typename NT>
YNN_ALWAYS_INLINE void store(float* ptr, f32x16 b, NT n) {
  if (n < f32x8::N) {
    store(ptr, b.v[0], n);
  } else {
    store(ptr, b.v[0]);
    store(ptr + f32x8::N, b.v[1], n - f32x8::N);
  }
}

struct accumulator_fp32 {
  static constexpr std::integral_constant<size_t, 4> N = {};
  static constexpr std::integral_constant<size_t, 16> K = {};

  f32x16 acc[N];

  accumulator_fp32() = default;

  YNN_ALWAYS_INLINE explicit accumulator_fp32(size_t k) {
    f32x16 zero(0.0f);
    for (size_t i = 0; i < N; ++i) {
      acc[i] = zero;
    }
  }

  template <typename NT, typename KT>
  YNN_ALWAYS_INLINE void reduce(const half* A, size_t a_stride_n, NT n, KT k) {
    const simd::vec<half, K> zero(0.0f);
    auto a_0 = load(offset_bytes(A, 0 * a_stride_n), zero, k);
    auto a_1 = 1 < n ? load(offset_bytes(A, 1 * a_stride_n), zero, k) : zero;
    auto a_2 = 2 < n ? load(offset_bytes(A, 2 * a_stride_n), zero, k) : zero;
    auto a_3 = 3 < n ? load(offset_bytes(A, 3 * a_stride_n), zero, k) : zero;
    acc[0] += a_0;
    acc[1] += a_1;
    acc[2] += a_2;
    acc[3] += a_3;
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

void sum_fp16_fp32_4x16_f16c(size_t n, size_t k3, size_t k2, size_t k1,
                             size_t a_stride_n, size_t a_stride_k3,
                             size_t a_stride_k2, const void* a, size_t,
                             void* c) {
  if (k1 == 1 && a_stride_n == sizeof(half)) {
    tiled_reduce<sum_accumulator_k1_1<f32x16>, half, float>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const half*>(a),
        /*C_stride_m=*/0, reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<accumulator_fp32, half, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const half*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

}  // namespace ynn
