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

#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/multi_vec.h"
#include "ynnpack/base/simd/x86_avx.h"
#include "ynnpack/base/simd/x86_avx512bw.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max_accumulator.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace simd {

using s32x16x2 = multi_vec<s32x16, 2>;
using s32x16x4 = multi_vec<s32x16, 4>;
using f32x16x16 = multi_vec<f32x16, 16>;
using bf16x32x8 = multi_vec<bf16x32, 8>;
using f16x32x8 = multi_vec<f16x32, 8>;

static s32x16x4& operator+=(s32x16x4& a, s8x64 b) {
  a.v[0] += s32x16{_mm512_cvtepi8_epi32(extract<0>(b, s8x16{}).v)};
  a.v[1] += s32x16{_mm512_cvtepi8_epi32(extract<1>(b, s8x16{}).v)};
  a.v[2] += s32x16{_mm512_cvtepi8_epi32(extract<2>(b, s8x16{}).v)};
  a.v[3] += s32x16{_mm512_cvtepi8_epi32(extract<3>(b, s8x16{}).v)};
  return a;
}

static s32x16x4& operator+=(s32x16x4& a, u8x64 b) {
  a.v[0] += s32x16{_mm512_cvtepu8_epi32(extract<0>(b, u8x16{}).v)};
  a.v[1] += s32x16{_mm512_cvtepu8_epi32(extract<1>(b, u8x16{}).v)};
  a.v[2] += s32x16{_mm512_cvtepu8_epi32(extract<2>(b, u8x16{}).v)};
  a.v[3] += s32x16{_mm512_cvtepu8_epi32(extract<3>(b, u8x16{}).v)};
  return a;
}

static f32x16& operator+=(f32x16& a, f16x16 b) {
  return a += convert(b, float{});
}

static f32x16x16& operator+=(f32x16x16& a, f16x32x8 b) {
  YNN_UNROLL
  for (size_t i = 0; i < 8; ++i) {
    a.v[2 * i + 0] += extract<0>(b.v[i], f16x16{});
    a.v[2 * i + 1] += extract<1>(b.v[i], f16x16{});
  }
  return a;
}

static s32x16 reduce_add(
    s32x16 a, s8x64 b, Identity /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  __m512i b2x = _mm512_maddubs_epi16(_mm512_set1_epi8(1), b.v);
  s32x16 b_s32(_mm512_madd_epi16(_mm512_set1_epi16(1), b2x));
  return a += b_s32;
}

static s32x16 reduce_add(
    s32x16 a, u8x64 b, Identity /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  __m512i b2x = _mm512_maddubs_epi16(b.v, _mm512_set1_epi8(1));
  s32x16 b_s32(_mm512_madd_epi16(_mm512_set1_epi16(1), b2x));
  return a += b_s32;
}

static s32x16x2 reduce_add(
    s32x16x2 a, s8x32 b, Square /*map_fn*/,
    std::integral_constant<size_t, 1>/*horizontal_factor*/) {
  // Convert int8 -> uint8 via abs first.
  __m256i abs_b = _mm256_abs_epi8(b.v);
  __m512i b_lo = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(abs_b));
  __m512i b_hi = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(abs_b, 1));

  // madd_epi16 works due to extra zeros from uint8 -> int32 conversion.
  a.v[0] += s32x16{_mm512_madd_epi16(b_lo, b_lo)};
  a.v[1] += s32x16{_mm512_madd_epi16(b_hi, b_hi)};
  return a;
}

static s32x16x2 reduce_add(
    s32x16x2 a, u8x32 b, Square /*map_fn*/,
    std::integral_constant<size_t, 1>/*horizontal_factor*/) {
  __m512i b_lo = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(b.v));
  __m512i b_hi = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(b.v, 1));

  // madd_epi16 works due to extra zeros from uint8 -> int32 conversion.
  a.v[0] += s32x16{_mm512_madd_epi16(b_lo, b_lo)};
  a.v[1] += s32x16{_mm512_madd_epi16(b_hi, b_hi)};
  return a;
}

static s32x16 reduce_add(
    s32x16 a, s8x32 b, Square /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  __m512i b_16 = _mm512_cvtepi8_epi16(b.v);
  return a += s32x16(_mm512_madd_epi16(b_16, b_16));
}

static s32x16 reduce_add(
    s32x16 a, u8x32 b, Square /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  __m512i b_16 = _mm512_cvtepu8_epi16(b.v);
  return a += s32x16(_mm512_madd_epi16(b_16, b_16));
}

static f32x16x16 reduce_add(
    f32x16x16 a, bf16x32x8 b, Identity /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  YNN_UNROLL
  for (int i = 0; i < 8; ++i) {
    a.v[2 * i + 0] += convert(extract<0>(b.v[i], bf16x16{}), float{});
    a.v[2 * i + 1] += convert(extract<1>(b.v[i], bf16x16{}), float{});
  }

  return a;
}

static f32x16 reduce_add(
    f32x16 a, bf16x32 b, Identity /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  __m512i mask = _mm512_set1_epi32(0xFFFF0000);
  f32x16 evens(_mm512_castsi512_ps(_mm512_slli_epi32(b.v, 16)));
  f32x16 odds(_mm512_castsi512_ps(_mm512_and_epi32(b.v, mask)));

  a += odds;
  a += evens;
  return a;
}

static f32x16x16 reduce_add(
    f32x16x16 a, bf16x32x8 b, Square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  YNN_UNROLL
  for (int i = 0; i < 8; ++i) {
    f32x16 lo = convert(extract<0>(b.v[i], bf16x16{}), float{});
    f32x16 hi = convert(extract<1>(b.v[i], bf16x16{}), float{});

    a.v[2 * i + 0].v = _mm512_fmadd_ps(lo.v, lo.v, a.v[2 * i + 0].v);
    a.v[2 * i + 1].v = _mm512_fmadd_ps(hi.v, hi.v, a.v[2 * i + 1].v);
  }

  return a;
}

static f32x16 reduce_add(
    f32x16 a, bf16x32 b, Square /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  __m512i mask = _mm512_set1_epi32(0xFFFF0000);
  __m512 evens = _mm512_castsi512_ps(_mm512_slli_epi32(b.v, 16));
  __m512 odds  = _mm512_castsi512_ps(_mm512_and_epi32(b.v, mask));

  a.v = _mm512_fmadd_ps(odds, odds, a.v);
  a.v = _mm512_fmadd_ps(evens, evens, a.v);
  return a;
}

static f32x16x16 reduce_add(
    f32x16x16 a, f16x32x8 b, Square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  YNN_UNROLL
  for (size_t i = 0; i < 8; ++i) {
    f32x16 lo = convert(extract<0>(b.v[i], f16x16{}), float{});
    f32x16 hi = convert(extract<1>(b.v[i], f16x16{}), float{});
    a.v[2 * i + 0].v = _mm512_fmadd_ps(lo.v, lo.v, a.v[2 * i + 0].v);
    a.v[2 * i + 1].v = _mm512_fmadd_ps(hi.v, hi.v, a.v[2 * i + 1].v);
  }
  return a;
}

static f32x16 reduce_add(
    f32x16 a, f16x16 b, Square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  f32x16 b_f32 = convert(b, float{});
  a.v = _mm512_fmadd_ps(b_f32.v, b_f32.v, a.v);
  return a;
}

}  // namespace simd

using simd::bf16x32;
using simd::bf16x32x8;
using simd::f16x32;
using simd::f16x32x8;
using simd::f32x16;
using simd::f32x16x16;
using simd::s16x32;
using simd::s32x16;
using simd::s32x16x2;
using simd::s32x16x4;
using simd::s32x8;
using simd::s8x32;
using simd::s8x64;
using simd::u8x32;
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

void sum_squared_int8_int32_avx512bw(size_t n, size_t k3, size_t k2, size_t k1,
                                     size_t a_stride_n, size_t a_stride_k3,
                                     size_t a_stride_k2, const void* a, size_t,
                                     void* c) {
  if (k1 == 1 && a_stride_n == sizeof(int8_t)) {
    tiled_reduce<sum_accumulator_k1_1<s8x32, s32x16x2, Square>, int8_t,
      int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const int8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x16, 32, Square>, int8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const int8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

void sum_squared_uint8_int32_avx512bw(size_t n, size_t k3, size_t k2, size_t k1,
                                      size_t a_stride_n, size_t a_stride_k3,
                                      size_t a_stride_k2, const void* a, size_t,
                                      void* c) {
  if (k1 == 1 && a_stride_n == sizeof(uint8_t)) {
    tiled_reduce<sum_accumulator_k1_1<u8x32, s32x16x2, Square>, uint8_t,
      int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x16, 32, Square>, uint8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

void sum_bf16_fp32_avx512bw(size_t n, size_t k3, size_t k2, size_t k1,
                            size_t a_stride_n, size_t a_stride_k3,
                            size_t a_stride_k2, const void* a, size_t,
                            void* c) {
  if (k1 == 1 && a_stride_n == sizeof(bfloat16)) {
    tiled_reduce<sum_accumulator_k1_1<bf16x32x8, f32x16x16>, bfloat16, float>(
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

void sum_squared_bf16_fp32_avx512bw(size_t n, size_t k3, size_t k2, size_t k1,
                                    size_t a_stride_n, size_t a_stride_k3,
                                    size_t a_stride_k2, const void* a, size_t,
                                    void* c) {
  if (k1 == 1 && a_stride_n == sizeof(bfloat16)) {
    tiled_reduce<sum_accumulator_k1_1<bf16x32x8, f32x16x16, Square>, bfloat16,
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

void sum_fp16_fp32_avx512bw(size_t n, size_t k3, size_t k2, size_t k1,
                            size_t a_stride_n, size_t a_stride_k3,
                            size_t a_stride_k2, const void* a, size_t,
                            void* c) {
  if (k1 == 1 && a_stride_n == sizeof(half)) {
    tiled_reduce<sum_accumulator_k1_1<f16x32x8, f32x16x16>, half, float>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const half*>(a),
        /*C_stride_m=*/0, reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<f32x16, 16>, half, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const half*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

void sum_squared_fp16_fp32_avx512bw(size_t n, size_t k3, size_t k2, size_t k1,
                                    size_t a_stride_n, size_t a_stride_k3,
                                    size_t a_stride_k2, const void* a, size_t,
                                    void* c) {
  if (k1 == 1 && a_stride_n == sizeof(half)) {
    tiled_reduce<sum_accumulator_k1_1<f16x32x8, f32x16x16, Square>, half,
                 float>(n, k3, k2, a_stride_k3, a_stride_k2,
                        reinterpret_cast<const half*>(a),
                        /*C_stride_m=*/0, reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<f32x16, 16, Square>, half, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const half*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

}  // namespace ynn
