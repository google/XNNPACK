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

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/x86_avx512.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max_accumulator.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"
#include "ynnpack/kernels/reduce/reduce.h"

namespace ynn {

namespace simd {

struct s32x16x2 {
  s32x16 v[2];

  using value_type = int32_t;
  static constexpr std::integral_constant<size_t, 32> N = {};

  s32x16x2() = default;
  explicit s32x16x2(int32_t x) : v{x, x} {};

  s32x16x2 operator+(s32x16x2 a) const {
    s32x16x2 res;
    res.v[0] = this->v[0] + a.v[0];
    res.v[1] = this->v[1] + a.v[1];
    return res;
  }
};

struct s32x16x4 {
  s32x16x2 v[2];

  using value_type = int32_t;
  static constexpr std::integral_constant<size_t, 64> N = {};

  s32x16x4() = default;
  explicit s32x16x4(int32_t x) : v{s32x16x2(x), s32x16x2(x)} {};

  s32x16x4 operator+(s32x16x4 a) const {
    s32x16x4 res;
    res.v[0] = this->v[0] + a.v[0];
    res.v[1] = this->v[1] + a.v[1];
    return res;
  }
};

static s32x16x2 load(const int32_t* ptr, s32x16x2, decltype(s32x16x2::N)) {
  s32x16x2 x;

  x.v[0] = load(ptr, s32x16{}, s32x16::N);
  x.v[1] = load(ptr + s32x16::N, s32x16{}, s32x16::N);

  return x;
}

static s32x16x2 load(const int32_t* ptr, s32x16x2, size_t n) {
  s32x16x2 x;

  if (n <= s32x16::N) {
    x.v[0] = load(ptr, s32x16{}, n);
    x.v[1] = 0;
  } else {
    x.v[0] = load(ptr, s32x16{}, s32x16::N);
    x.v[1] = load(ptr + s32x16::N, s32x16{}, n - s32x16::N);
  }

  return x;
}

static s32x16x4 load(const int32_t* ptr, s32x16x4, decltype(s32x16x4::N)) {
  s32x16x4 x;

  x.v[0] = load(ptr, s32x16x2{}, s32x16x2::N);
  x.v[1] = load(ptr + s32x16x2::N, s32x16x2{}, s32x16x2::N);

  return x;
}

static s32x16x4 load(const int32_t* ptr, s32x16x4, size_t n) {
  s32x16x4 x;

  if (n <= s32x16x2::N) {
    x.v[0] = load(ptr, s32x16x2{}, n);
    x.v[1] = s32x16x2(0);
  } else {
    x.v[0] = load(ptr, s32x16x2{}, s32x16x2::N);
    x.v[1] = load(ptr + s32x16x2::N, s32x16x2{}, n - s32x16x2::N);
  }

  return x;
}

static void store(int32_t* ptr, s32x16x2 b, decltype(s32x16x2::N)) {
  store(ptr, b.v[0]);
  store(ptr + s32x16::N, b.v[1]);
}

static void store(int32_t* ptr, s32x16x2 b, size_t n) {
  if (n <= s32x16::N) {
    store(ptr, b.v[0], n);
  } else {
    store(ptr, b.v[0]);
    store(ptr + s32x16::N, b.v[1], n - s32x16::N);
  }
}

static void store(int32_t* ptr, s32x16x4 b, decltype(s32x16x4::N)) {
  store(ptr, b.v[0], s32x16x2::N);
  store(ptr + s32x16x2::N, b.v[1], s32x16x2::N);
}

static void store(int32_t* ptr, s32x16x4 b, size_t n) {
  if (n <= s32x16x2::N) {
    store(ptr, b.v[0], n);
  } else {
    store(ptr, b.v[0], s32x16x2::N);
    store(ptr + s32x16x2::N, b.v[1], n - s32x16x2::N);
  }
}

static s32x16x2& operator+=(s32x16x2& a, s8x32 b) {
  s32x16 b_0(_mm512_cvtepi8_epi32(_mm256_extractf128_si256(b.v, 0)));
  s32x16 b_1(_mm512_cvtepi8_epi32(_mm256_extractf128_si256(b.v, 1)));

  a.v[0] += b_0;
  a.v[1] += b_1;
  return a;
}

static s32x16x2& operator+=(s32x16x2& a, u8x32 b) {
  s32x16 b_0(_mm512_cvtepu8_epi32(_mm256_extractf128_si256(b.v, 0)));
  s32x16 b_1(_mm512_cvtepu8_epi32(_mm256_extractf128_si256(b.v, 1)));

  a.v[0] += b_0;
  a.v[1] += b_1;
  return a;
}

static s32x16x4& operator+=(s32x16x4& a, s8x64 b) {
  a.v[0] += extract<0>(b, s8x32{});
  a.v[1] += extract<1>(b, s8x32{});
  return a;
}

static s32x16x4& operator+=(s32x16x4& a, u8x64 b) {
  a.v[0] += extract<0>(b, u8x32{});
  a.v[1] += extract<1>(b, u8x32{});
  return a;
}

static s32x16& reduce_add(
    s32x16& a, s8x64 b,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  __m512i b2x = _mm512_maddubs_epi16(_mm512_set1_epi8(1), b.v);
  s32x16 b_s32(_mm512_madd_epi16(_mm512_set1_epi16(1), b2x));
  return a += b_s32;
}

static s32x16& reduce_add(
    s32x16& a, u8x64 b,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  __m512i b2x = _mm512_maddubs_epi16(b.v, _mm512_set1_epi8(1));
  s32x16 b_s32(_mm512_madd_epi16(_mm512_set1_epi16(1), b2x));
  return a += b_s32;
}

}  // namespace simd

using simd::s32x8;
using simd::s32x16;
using simd::s32x16x4;
using simd::bf16x32;
using simd::f16x32;
using simd::s16x32;
using simd::s8x64;
using simd::u8x64;

using f16x32_rvar = float16_wrapper<f16x32, s16x32>;
using bf16x32_rvar = float16_wrapper<bf16x32, s16x32>;

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

void sum_int8_int32_avx512bw(size_t n, size_t k3, size_t k2, size_t k1,
                             size_t a_stride_n, size_t a_stride_k3,
                             size_t a_stride_k2, const void* a, size_t,
                             void* c) {
  if (k1 == 1 && a_stride_n == sizeof(int8_t)) {
    tiled_reduce<sum_accumulator_k1_1<s8x64, s32x16x4>, int8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const int8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x16, 64>, int8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const int8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

void sum_uint8_int32_avx512bw(size_t n, size_t k3, size_t k2, size_t k1,
                              size_t a_stride_n, size_t a_stride_k3,
                              size_t a_stride_k2, const void* a, size_t,
                              void* c) {
  if (k1 == 1 && a_stride_n == sizeof(uint8_t)) {
    tiled_reduce<sum_accumulator_k1_1<u8x64, s32x16x4>, uint8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x16, 64>, uint8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

}  // namespace ynn
