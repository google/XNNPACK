// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_avx512.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max_accumulator.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace simd {

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

static s32x64 reduce_add(
    s32x64 a, u8x64 b, Square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  s32x64 b_s32 = convert(b, int32_t{});

  // madd_epi16 works due to extra zeros from uint8 -> int32 conversion.
  a[0][0] += s32x16{_mm512_madd_epi16(b_s32[0][0].v, b_s32[0][0].v)};
  a[0][1] += s32x16{_mm512_madd_epi16(b_s32[0][1].v, b_s32[0][1].v)};
  a[1][0] += s32x16{_mm512_madd_epi16(b_s32[1][0].v, b_s32[1][0].v)};
  a[1][1] += s32x16{_mm512_madd_epi16(b_s32[1][1].v, b_s32[1][1].v)};
  return a;
}

static s32x64 reduce_add(s32x64 a, s8x64 b, Square map_fn,
                         std::integral_constant<size_t, 1> horizontal_factor) {
  // We're squaring, we can take the absolute value and use the unsigned reduce.
  return reduce_add(a, abs(b), map_fn, horizontal_factor);
}

static s32x16 reduce_add(
    s32x16 a, s8x64 b, Square /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  __m512i b_even = _mm512_and_si512(b.v, _mm512_set1_epi16(0x00FF));
  // Negative values won't be sign extended by the mask above. Sign extension is
  // expensive, and since we're squaring, we can just take the absolute value
  // instead.
  b_even = _mm512_abs_epi8(b_even);
  __m512i b_odd = _mm512_srai_epi16(b.v, 8);
  a += s32x16(_mm512_madd_epi16(b_even, b_even));
  a += s32x16(_mm512_madd_epi16(b_odd, b_odd));
  return a;
}

static s32x16 reduce_add(
    s32x16 a, u8x64 b, Square /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  __m512i b_even = _mm512_and_si512(b.v, _mm512_set1_epi16(0x00FF));
  __m512i b_odd = _mm512_srli_epi16(b.v, 8);
  a += s32x16(_mm512_madd_epi16(b_even, b_even));
  a += s32x16(_mm512_madd_epi16(b_odd, b_odd));
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

static f32x32 reduce_add(
    f32x32 a, bf16x32 b, Square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  f32x32 b_f32 = convert(b, float{});
  return fma(b_f32, b_f32, a);
}

static f32x16 reduce_add(
    f32x16 a, bf16x32 b, Square /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  __m512i mask = _mm512_set1_epi32(0xFFFF0000);
  f32x16 evens{_mm512_castsi512_ps(_mm512_slli_epi32(b.v, 16))};
  f32x16 odds{_mm512_castsi512_ps(_mm512_and_epi32(b.v, mask))};

  a = fma(odds, odds, a);
  a = fma(evens, evens, a);
  return a;
}

static f32x32 reduce_add(
    f32x32 a, f16x32 b, Square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  f32x32 b_f32 = convert(b, float{});
  return fma(b_f32, b_f32, a);
}

static f32x16 reduce_add(
    f32x16 a, f16x16 b, Square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  f32x16 b_f32 = convert(b, float{});
  return fma(b_f32, b_f32, a);
}

}  // namespace simd

using simd::bf16x32;
using simd::f16x32;
using simd::f32x16;
using simd::f32x32;
using simd::s16x32;
using simd::s32x16;
using simd::s32x32;
using simd::s32x64;
using simd::s32x8;
using simd::s8x16;
using simd::s8x32;
using simd::s8x64;
using simd::u8x16;
using simd::u8x32;
using simd::u8x64;

using f16x32_rvar = float16_wrapper<f16x32, s16x32>;
using bf16x32_rvar = float16_wrapper<bf16x32, s16x32>;

MIN_MAX_KERNEL(min_max_fp32_4x16_avx512, f32x16, f32x16, float, 16);
MIN_MAX_KERNEL(min_max_bf16_4x32_avx512, bf16x32_rvar, bf16x32_rvar, bfloat16,
               32);
MIN_MAX_KERNEL(min_max_fp16_4x32_avx512, f16x32_rvar, f16x32_rvar, half, 32);
MIN_MAX_KERNEL(min_max_uint8_4x64_avx512, u8x64, u8x64, uint8_t, 64);
MIN_MAX_KERNEL(min_max_int8_4x64_avx512, s8x64, s8x64, int8_t, 64);

MIN_MAX_KERNEL(min_fp32_4x16_avx512, f32x16, dummy_t, float, 16);
MIN_MAX_KERNEL(min_bf16_4x32_avx512, bf16x32_rvar, dummy_t, bfloat16, 32);
MIN_MAX_KERNEL(min_fp16_4x32_avx512, f16x32_rvar, dummy_t, half, 32);
MIN_MAX_KERNEL(min_uint8_4x64_avx512, u8x64, dummy_t, uint8_t, 64);
MIN_MAX_KERNEL(min_int8_4x64_avx512, s8x64, dummy_t, int8_t, 64);

MIN_MAX_KERNEL(max_fp32_4x16_avx512, dummy_t, f32x16, float, 16);
MIN_MAX_KERNEL(max_bf16_4x32_avx512, dummy_t, bf16x32_rvar, bfloat16, 32);
MIN_MAX_KERNEL(max_fp16_4x32_avx512, dummy_t, f16x32_rvar, half, 32);
MIN_MAX_KERNEL(max_uint8_4x64_avx512, dummy_t, u8x64, uint8_t, 64);
MIN_MAX_KERNEL(max_int8_4x64_avx512, dummy_t, s8x64, int8_t, 64);

void sum_fp32_avx512(size_t n, size_t k3, size_t k2, size_t k1,
                      size_t a_stride_n, size_t a_stride_k3,
                      size_t a_stride_k2, const void* a, size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(float)) {
    stream_reduce<sum_accumulator_k1_1<f32x16>, float, float>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const float*>(a),
        /*C_stride_m=*/0, reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_fp32<>, float, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const float*>(a),
        /*C_stride_m=*/0, reinterpret_cast<float*>(c));
  }
}

void sum_squared_fp32_avx512(size_t n, size_t k3, size_t k2, size_t k1,
                              size_t a_stride_n, size_t a_stride_k3,
                              size_t a_stride_k2, const void* a, size_t,
                              void* c) {
  if (k1 == 1 && a_stride_n == sizeof(float)) {
    stream_reduce<sum_accumulator_k1_1<f32x16, Square>, float, float>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const float*>(a),
        /*C_stride_m=*/0, reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_fp32<1, Square>, float, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const float*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

void sum_int8_int32_avx512(size_t n, size_t k3, size_t k2, size_t k1,
                             size_t a_stride_n, size_t a_stride_k3,
                             size_t a_stride_k2, const void* a, size_t,
                             void* c) {
  if (k1 == 1 && a_stride_n == sizeof(int8_t)) {
    stream_reduce<sum_accumulator_k1_1<s32x64>, int8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const int8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x16, 64>, int8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const int8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

void sum_uint8_int32_avx512(size_t n, size_t k3, size_t k2, size_t k1,
                              size_t a_stride_n, size_t a_stride_k3,
                              size_t a_stride_k2, const void* a, size_t,
                              void* c) {
  if (k1 == 1 && a_stride_n == sizeof(uint8_t)) {
    stream_reduce<sum_accumulator_k1_1<s32x64>, uint8_t, int32_t>(
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

void sum_squared_int8_int32_avx512(size_t n, size_t k3, size_t k2, size_t k1,
                                     size_t a_stride_n, size_t a_stride_k3,
                                     size_t a_stride_k2, const void* a, size_t,
                                     void* c) {
  if (k1 == 1 && a_stride_n == sizeof(int8_t)) {
    stream_reduce<sum_accumulator_k1_1<s32x64, Square>, int8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const int8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x16, 64, Square>, int8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const int8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

void sum_squared_uint8_int32_avx512(size_t n, size_t k3, size_t k2, size_t k1,
                                      size_t a_stride_n, size_t a_stride_k3,
                                      size_t a_stride_k2, const void* a, size_t,
                                      void* c) {
  if (k1 == 1 && a_stride_n == sizeof(uint8_t)) {
    stream_reduce<sum_accumulator_k1_1<s32x64, Square>, uint8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x16, 64, Square>, uint8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

void sum_bf16_fp32_avx512(size_t n, size_t k3, size_t k2, size_t k1,
                            size_t a_stride_n, size_t a_stride_k3,
                            size_t a_stride_k2, const void* a, size_t,
                            void* c) {
  if (k1 == 1 && a_stride_n == sizeof(bfloat16)) {
    stream_reduce<sum_accumulator_k1_1<f32x32>, bfloat16, float>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_fp32<2>, bfloat16, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

void sum_squared_bf16_fp32_avx512(size_t n, size_t k3, size_t k2, size_t k1,
                                    size_t a_stride_n, size_t a_stride_k3,
                                    size_t a_stride_k2, const void* a, size_t,
                                    void* c) {
  if (k1 == 1 && a_stride_n == sizeof(bfloat16)) {
    stream_reduce<sum_accumulator_k1_1<f32x32, Square>, bfloat16, float>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_fp32<2, Square>, bfloat16, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

void sum_fp16_fp32_avx512(size_t n, size_t k3, size_t k2, size_t k1,
                            size_t a_stride_n, size_t a_stride_k3,
                            size_t a_stride_k2, const void* a, size_t,
                            void* c) {
  if (k1 == 1 && a_stride_n == sizeof(half)) {
    stream_reduce<sum_accumulator_k1_1<f32x32>, half, float>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const half*>(a),
        /*C_stride_m=*/0, reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_fp32<>, half, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const half*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

void sum_squared_fp16_fp32_avx512(size_t n, size_t k3, size_t k2, size_t k1,
                                    size_t a_stride_n, size_t a_stride_k3,
                                    size_t a_stride_k2, const void* a, size_t,
                                    void* c) {
  if (k1 == 1 && a_stride_n == sizeof(half)) {
    stream_reduce<sum_accumulator_k1_1<f32x32, Square>, half, float>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const half*>(a),
        /*C_stride_m=*/0, reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_fp32<1, Square>, half, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const half*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

}  // namespace ynn
