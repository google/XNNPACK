// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_AVX512_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_AVX512_H_

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>

#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/simd/x86_avx2_base.h"  // IWYU pragma: export

namespace ynn {

namespace simd {

// See vec.h for architecture independent comments.

namespace internal {

YNN_ALWAYS_INLINE __m256 lo(__m512 x) { return _mm512_castps512_ps256(x); }
YNN_ALWAYS_INLINE __m256 hi(__m512 x) {
  return _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(x), 1));
}

YNN_ALWAYS_INLINE __m256d lo(__m512d x) { return _mm512_castpd512_pd256(x); }
YNN_ALWAYS_INLINE __m256d hi(__m512d x) { return _mm512_extractf64x4_pd(x, 1); }

YNN_ALWAYS_INLINE __m256i lo(__m512i x) { return _mm512_castsi512_si256(x); }
YNN_ALWAYS_INLINE __m256i hi(__m512i x) {
  return _mm512_extracti64x4_epi64(x, 1);
}

YNN_ALWAYS_INLINE __m512 concat(__m256 lo, __m256 hi) {
  return _mm512_castpd_ps(
      _mm512_insertf64x4(_mm512_castps_pd(_mm512_castps256_ps512(lo)),
                         _mm256_castps_pd(hi), 1));
}

YNN_ALWAYS_INLINE __m512d concat(__m256d lo, __m256d hi) {
  return _mm512_insertf64x4(_mm512_castpd256_pd512(lo), hi, 1);
}

YNN_ALWAYS_INLINE __m512i concat(__m256i lo, __m256i hi) {
  return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

}  // namespace internal

template <>
struct vec<double, 8> {
  using value_type = double;
  static constexpr std::integral_constant<size_t, 8> N = {};

  vec() = default;
  explicit vec(__m512d v) : v(v) {}
  vec(f64x4 lo, f64x4 hi) : v(internal::concat(lo.v, hi.v)) {}
  vec(double x) : v(_mm512_set1_pd(x)) {}  // NOLINT

  __m512d v;
};

template <>
struct vec<float, 16> {
  using value_type = float;
  static constexpr std::integral_constant<size_t, 16> N = {};

  vec() = default;
  explicit vec(__m512 v) : v(v) {}
  vec(f32x8 lo, f32x8 hi) : v(internal::concat(lo.v, hi.v)) {}
  vec(float x) : v(_mm512_set1_ps(x)) {}  // NOLINT

  __m512 v;
};

template <>
struct vec<uint32_t, 16> {
  using value_type = uint32_t;
  static constexpr std::integral_constant<size_t, 16> N = {};

  vec() = default;
  explicit vec(__m512i v) : v(v) {}
  vec(u32x8 lo, u32x8 hi) : v(internal::concat(lo.v, hi.v)) {}
  vec(uint32_t x) : v(_mm512_set1_epi32(x)) {}  // NOLINT

  __m512i v;
};

template <>
struct vec<int32_t, 16> {
  using value_type = int32_t;
  static constexpr std::integral_constant<size_t, 16> N = {};

  vec() = default;
  explicit vec(__m512i v) : v(v) {}
  vec(s32x8 lo, s32x8 hi) : v(internal::concat(lo.v, hi.v)) {}
  vec(int32_t x) : v(_mm512_set1_epi32(x)) {}  // NOLINT

  __m512i v;
};

template <>
struct vec<bfloat16, 32> {
  using value_type = bfloat16;
  static constexpr std::integral_constant<size_t, 32> N = {};

  vec() = default;
  explicit vec(__m512i v) : v(v) {}
  vec(bf16x16 lo, bf16x16 hi) : v(internal::concat(lo.v, hi.v)) {}
  vec(bfloat16 x) : v(_mm512_set1_epi16(x.to_bits())) {}  // NOLINT

  __m512i v;
};

template <>
struct vec<half, 32> {
  using value_type = half;
  static constexpr std::integral_constant<size_t, 32> N = {};

  vec() = default;
  explicit vec(__m512i v) : v(v) {}
  vec(f16x16 lo, f16x16 hi) : v(internal::concat(lo.v, hi.v)) {}
  vec(half x) : v(_mm512_set1_epi16(x.to_bits())) {}  // NOLINT

  __m512i v;
};

template <>
struct vec<uint16_t, 32> {
  using value_type = uint16_t;
  static constexpr std::integral_constant<size_t, 32> N = {};

  vec() = default;
  explicit vec(__m512i v) : v(v) {}
  vec(u16x16 lo, u16x16 hi) : v(internal::concat(lo.v, hi.v)) {}
  vec(uint16_t x) : v(_mm512_set1_epi16(x)) {}  // NOLINT

  __m512i v;
};

template <>
struct vec<int16_t, 32> {
  using value_type = int16_t;
  static constexpr std::integral_constant<size_t, 32> N = {};

  vec() = default;
  explicit vec(__m512i v) : v(v) {}
  vec(s16x16 lo, s16x16 hi) : v(internal::concat(lo.v, hi.v)) {}
  vec(int16_t x) : v(_mm512_set1_epi16(x)) {}  // NOLINT

  __m512i v;
};

template <>
struct vec<uint8_t, 64> {
  using value_type = uint8_t;
  static constexpr std::integral_constant<size_t, 64> N = {};

  vec() = default;
  explicit vec(__m512i v) : v(v) {}
  vec(u8x32 lo, u8x32 hi) : v(internal::concat(lo.v, hi.v)) {}
  vec(uint8_t x) : v(_mm512_set1_epi8(x)) {}  // NOLINT

  __m512i v;
};

template <>
struct vec<int8_t, 64> {
  using value_type = int8_t;
  static constexpr std::integral_constant<size_t, 64> N = {};

  vec() = default;
  explicit vec(__m512i v) : v(v) {}
  vec(s8x32 lo, s8x32 hi) : v(internal::concat(lo.v, hi.v)) {}
  vec(int8_t x) : v(_mm512_set1_epi8(x)) {}  // NOLINT

  __m512i v;
};

using f64x8 = vec<double, 8>;
using f32x16 = vec<float, 16>;
using u32x16 = vec<uint32_t, 16>;
using s32x16 = vec<int32_t, 16>;
using bf16x32 = vec<bfloat16, 32>;
using f16x32 = vec<half, 32>;
using u16x32 = vec<uint16_t, 32>;
using s16x32 = vec<int16_t, 32>;
using u8x64 = vec<uint8_t, 64>;
using s8x64 = vec<int8_t, 64>;
using f32x64 = vec<float, 64>;

YNN_ALWAYS_INLINE f64x4 lo(f64x8 x) { return f64x4{internal::lo(x.v)}; }
YNN_ALWAYS_INLINE f64x4 hi(f64x8 x) { return f64x4{internal::hi(x.v)}; }
YNN_ALWAYS_INLINE f32x8 lo(f32x16 x) { return f32x8{internal::lo(x.v)}; }
YNN_ALWAYS_INLINE f32x8 hi(f32x16 x) { return f32x8{internal::hi(x.v)}; }
YNN_ALWAYS_INLINE u32x8 lo(u32x16 x) { return u32x8{internal::lo(x.v)}; }
YNN_ALWAYS_INLINE u32x8 hi(u32x16 x) { return u32x8{internal::hi(x.v)}; }
YNN_ALWAYS_INLINE s32x8 lo(s32x16 x) { return s32x8{internal::lo(x.v)}; }
YNN_ALWAYS_INLINE s32x8 hi(s32x16 x) { return s32x8{internal::hi(x.v)}; }
YNN_ALWAYS_INLINE bf16x16 lo(bf16x32 x) { return bf16x16{internal::lo(x.v)}; }
YNN_ALWAYS_INLINE bf16x16 hi(bf16x32 x) { return bf16x16{internal::hi(x.v)}; }
YNN_ALWAYS_INLINE f16x16 lo(f16x32 x) { return f16x16{internal::lo(x.v)}; }
YNN_ALWAYS_INLINE f16x16 hi(f16x32 x) { return f16x16{internal::hi(x.v)}; }
YNN_ALWAYS_INLINE u16x16 lo(u16x32 x) { return u16x16{internal::lo(x.v)}; }
YNN_ALWAYS_INLINE u16x16 hi(u16x32 x) { return u16x16{internal::hi(x.v)}; }
YNN_ALWAYS_INLINE s16x16 lo(s16x32 x) { return s16x16{internal::lo(x.v)}; }
YNN_ALWAYS_INLINE s16x16 hi(s16x32 x) { return s16x16{internal::hi(x.v)}; }
YNN_ALWAYS_INLINE u8x32 lo(u8x64 x) { return u8x32{internal::lo(x.v)}; }
YNN_ALWAYS_INLINE u8x32 hi(u8x64 x) { return u8x32{internal::hi(x.v)}; }
YNN_ALWAYS_INLINE s8x32 lo(s8x64 x) { return s8x32{internal::lo(x.v)}; }
YNN_ALWAYS_INLINE s8x32 hi(s8x64 x) { return s8x32{internal::hi(x.v)}; }

YNN_ALWAYS_INLINE f64x8 load_aligned(const double* ptr, decltype(f64x8::N),
                                     f64x8 = {}) {
  return f64x8{_mm512_load_pd(ptr)};
}
YNN_ALWAYS_INLINE f32x16 load_aligned(const float* ptr, decltype(f32x16::N),
                                      f32x16 = {}) {
  return f32x16{_mm512_load_ps(ptr)};
}
YNN_ALWAYS_INLINE s32x16 load_aligned(const int32_t* ptr, decltype(s32x16::N),
                                      s32x16 = {}) {
  return s32x16{_mm512_load_si512(reinterpret_cast<const __m512i*>(ptr))};
}
YNN_ALWAYS_INLINE bf16x32 load_aligned(const bfloat16* ptr,
                                       decltype(bf16x32::N), bf16x32 = {}) {
  return bf16x32{_mm512_load_si512(reinterpret_cast<const __m512i*>(ptr))};
}
YNN_ALWAYS_INLINE f16x32 load_aligned(const half* ptr, decltype(f16x32::N),
                                      f16x32 = {}) {
  return f16x32{_mm512_load_si512(reinterpret_cast<const __m512i*>(ptr))};
}
YNN_ALWAYS_INLINE s16x32 load_aligned(const int16_t* ptr, decltype(s16x32::N),
                                      s16x32 = {}) {
  return s16x32{_mm512_load_si512(reinterpret_cast<const __m512i*>(ptr))};
}
YNN_ALWAYS_INLINE u8x64 load_aligned(const uint8_t* ptr, decltype(u8x64::N),
                                     u8x64 = {}) {
  return u8x64{_mm512_load_si512(reinterpret_cast<const __m512i*>(ptr))};
}
YNN_ALWAYS_INLINE s8x64 load_aligned(const int8_t* ptr, decltype(s8x64::N),
                                     s8x64 = {}) {
  return s8x64{_mm512_load_si512(reinterpret_cast<const __m512i*>(ptr))};
}

YNN_ALWAYS_INLINE void store_aligned(double* ptr, f64x8 b,
                                     decltype(f64x8::N) = {}) {
  _mm512_store_pd(ptr, b.v);
}
YNN_ALWAYS_INLINE void store_aligned(float* ptr, f32x16 b,
                                     decltype(f32x16::N) = {}) {
  _mm512_store_ps(ptr, b.v);
}
YNN_ALWAYS_INLINE void store_aligned(uint32_t* ptr, u32x16 b,
                                     decltype(u32x16::N) = {}) {
  _mm512_store_si512(reinterpret_cast<__m512i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store_aligned(int32_t* ptr, s32x16 b,
                                     decltype(s32x16::N) = {}) {
  _mm512_store_si512(reinterpret_cast<__m512i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store_aligned(bfloat16* ptr, bf16x32 b,
                                     decltype(bf16x32::N) = {}) {
  _mm512_store_si512(reinterpret_cast<__m512i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store_aligned(half* ptr, f16x32 b,
                                     decltype(f16x32::N) = {}) {
  _mm512_store_si512(reinterpret_cast<__m512i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store_aligned(uint16_t* ptr, u16x32 b,
                                     decltype(u16x32::N) = {}) {
  _mm512_store_si512(reinterpret_cast<__m512i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store_aligned(int16_t* ptr, s16x32 b,
                                     decltype(s16x32::N) = {}) {
  _mm512_store_si512(reinterpret_cast<__m512i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store_aligned(uint8_t* ptr, u8x64 b,
                                     decltype(u8x64::N) = {}) {
  _mm512_store_si512(reinterpret_cast<__m512i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store_aligned(int8_t* ptr, s8x64 b,
                                     decltype(s8x64::N) = {}) {
  _mm512_store_si512(reinterpret_cast<__m512i*>(ptr), b.v);
}

YNN_ALWAYS_INLINE f64x8 load(const double* ptr, decltype(f64x8::N),
                             f64x8 = {}) {
  return f64x8{_mm512_loadu_pd(ptr)};
}
YNN_ALWAYS_INLINE f32x16 load(const float* ptr, decltype(f32x16::N),
                              f32x16 = {}) {
  return f32x16{_mm512_loadu_ps(ptr)};
}
YNN_ALWAYS_INLINE s32x16 load(const int32_t* ptr, decltype(s32x16::N),
                              s32x16 = {}) {
  return s32x16{_mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr))};
}
YNN_ALWAYS_INLINE bf16x32 load(const bfloat16* ptr, decltype(bf16x32::N),
                               bf16x32 = {}) {
  return bf16x32{_mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr))};
}
YNN_ALWAYS_INLINE f16x32 load(const half* ptr, decltype(f16x32::N),
                              f16x32 = {}) {
  return f16x32{_mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr))};
}
YNN_ALWAYS_INLINE s16x32 load(const int16_t* ptr, decltype(s16x32::N),
                              s16x32 = {}) {
  return s16x32{_mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr))};
}
YNN_ALWAYS_INLINE u8x64 load(const uint8_t* ptr, decltype(u8x64::N),
                             u8x64 = {}) {
  return u8x64{_mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr))};
}
YNN_ALWAYS_INLINE s8x64 load(const int8_t* ptr, decltype(s8x64::N),
                             s8x64 = {}) {
  return s8x64{_mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr))};
}

YNN_ALWAYS_INLINE void store(double* ptr, f64x8 b, decltype(f64x8::N) = {}) {
  _mm512_storeu_pd(ptr, b.v);
}
YNN_ALWAYS_INLINE void store(float* ptr, f32x16 b, decltype(f32x16::N) = {}) {
  _mm512_storeu_ps(ptr, b.v);
}
YNN_ALWAYS_INLINE void store(uint32_t* ptr, u32x16 b,
                             decltype(u32x16::N) = {}) {
  _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x16 b, decltype(s32x16::N) = {}) {
  _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(bfloat16* ptr, bf16x32 b,
                             decltype(bf16x32::N) = {}) {
  _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(half* ptr, f16x32 b, decltype(f16x32::N) = {}) {
  _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(uint16_t* ptr, u16x32 b,
                             decltype(u16x32::N) = {}) {
  _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(int16_t* ptr, s16x32 b, decltype(s16x32::N) = {}) {
  _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x64 b, decltype(u8x64::N) = {}) {
  _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(int8_t* ptr, s8x64 b, decltype(s8x64::N) = {}) {
  _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), b.v);
}

namespace internal {

YNN_ALWAYS_INLINE __mmask8 mask_x2(size_t n) {
  assert(n <= 2);
  return _cvtu32_mask8((uint32_t)((1 << n) - 1));
}

YNN_ALWAYS_INLINE __mmask8 mask_x4(size_t n) {
  assert(n <= 4);
  return _cvtu32_mask8((uint32_t)((1 << n) - 1));
}

YNN_ALWAYS_INLINE __mmask8 mask_x8(size_t n) {
  assert(n <= 8);
  return _cvtu32_mask8((uint32_t)((1 << n) - 1));
}

YNN_ALWAYS_INLINE __mmask16 mask_x16(size_t n) {
  assert(n <= 16);
  return _cvtu32_mask16((uint32_t)((1 << n) - 1));
}

YNN_ALWAYS_INLINE __mmask32 mask_x32(size_t n) {
  assert(n <= 32);
  return (1ULL << n) - 1;
}

YNN_ALWAYS_INLINE __mmask64 mask_x64(size_t n) {
  assert(n <= 64);
  return _cvtu64_mask64(((1ull << n) - 1));
}

}  // namespace internal

YNN_ALWAYS_INLINE f64x8 load(const double* ptr, size_t n, f64x8 src) {
  return f64x8{_mm512_mask_loadu_pd(src.v, internal::mask_x8(n), ptr)};
}
YNN_ALWAYS_INLINE f32x16 load(const float* ptr, size_t n, f32x16 src) {
  return f32x16{_mm512_mask_loadu_ps(src.v, internal::mask_x16(n), ptr)};
}
YNN_ALWAYS_INLINE s32x16 load(const int32_t* ptr, size_t n, s32x16 src) {
  return s32x16{_mm512_mask_loadu_epi32(src.v, internal::mask_x16(n), ptr)};
}
YNN_ALWAYS_INLINE bf16x32 load(const bfloat16* ptr, size_t n, bf16x32 src) {
  return bf16x32{_mm512_mask_loadu_epi16(src.v, internal::mask_x32(n), ptr)};
}
YNN_ALWAYS_INLINE f16x32 load(const half* ptr, size_t n, f16x32 src) {
  return f16x32{_mm512_mask_loadu_epi16(src.v, internal::mask_x32(n), ptr)};
}
YNN_ALWAYS_INLINE s16x32 load(const int16_t* ptr, size_t n, s16x32 src) {
  return s16x32{_mm512_mask_loadu_epi16(src.v, internal::mask_x32(n), ptr)};
}
YNN_ALWAYS_INLINE u8x64 load(const uint8_t* ptr, size_t n, u8x64 src) {
  return u8x64{_mm512_mask_loadu_epi8(src.v, internal::mask_x64(n), ptr)};
}
YNN_ALWAYS_INLINE s8x64 load(const int8_t* ptr, size_t n, s8x64 src) {
  return s8x64{_mm512_mask_loadu_epi8(src.v, internal::mask_x64(n), ptr)};
}

YNN_ALWAYS_INLINE f64x8 load(const double* ptr, size_t n, zeros<8>) {
  return f64x8{_mm512_maskz_loadu_pd(internal::mask_x8(n), ptr)};
}
YNN_ALWAYS_INLINE f32x16 load(const float* ptr, size_t n, zeros<16>) {
  return f32x16{_mm512_maskz_loadu_ps(internal::mask_x16(n), ptr)};
}
YNN_ALWAYS_INLINE s32x16 load(const int32_t* ptr, size_t n, zeros<16>) {
  return s32x16{_mm512_maskz_loadu_epi32(internal::mask_x16(n), ptr)};
}
YNN_ALWAYS_INLINE bf16x32 load(const bfloat16* ptr, size_t n, zeros<32>) {
  return bf16x32{_mm512_maskz_loadu_epi16(internal::mask_x32(n), ptr)};
}
YNN_ALWAYS_INLINE f16x32 load(const half* ptr, size_t n, zeros<32>) {
  return f16x32{_mm512_maskz_loadu_epi16(internal::mask_x32(n), ptr)};
}
YNN_ALWAYS_INLINE s16x32 load(const int16_t* ptr, size_t n, zeros<32>) {
  return s16x32{_mm512_maskz_loadu_epi16(internal::mask_x32(n), ptr)};
}
YNN_ALWAYS_INLINE u8x64 load(const uint8_t* ptr, size_t n, zeros<64>) {
  return u8x64{_mm512_maskz_loadu_epi8(internal::mask_x64(n), ptr)};
}
YNN_ALWAYS_INLINE s8x64 load(const int8_t* ptr, size_t n, zeros<64>) {
  return s8x64{_mm512_maskz_loadu_epi8(internal::mask_x64(n), ptr)};
}

YNN_ALWAYS_INLINE f64x8 load(const double* ptr, size_t n, undef<8>) {
  return f64x8{_mm512_maskz_loadu_pd(internal::mask_x8(n), ptr)};
}
YNN_ALWAYS_INLINE f32x16 load(const float* ptr, size_t n, undef<16>) {
  return f32x16{_mm512_maskz_loadu_ps(internal::mask_x16(n), ptr)};
}
YNN_ALWAYS_INLINE s32x16 load(const int32_t* ptr, size_t n, undef<16>) {
  return s32x16{_mm512_maskz_loadu_epi32(internal::mask_x16(n), ptr)};
}
YNN_ALWAYS_INLINE bf16x32 load(const bfloat16* ptr, size_t n, undef<32>) {
  return bf16x32{_mm512_maskz_loadu_epi16(internal::mask_x32(n), ptr)};
}
YNN_ALWAYS_INLINE f16x32 load(const half* ptr, size_t n, undef<32>) {
  return f16x32{_mm512_maskz_loadu_epi16(internal::mask_x32(n), ptr)};
}
YNN_ALWAYS_INLINE s16x32 load(const int16_t* ptr, size_t n, undef<32>) {
  return s16x32{_mm512_maskz_loadu_epi16(internal::mask_x32(n), ptr)};
}
YNN_ALWAYS_INLINE u8x64 load(const uint8_t* ptr, size_t n, undef<64>) {
  return u8x64{_mm512_maskz_loadu_epi8(internal::mask_x64(n), ptr)};
}
YNN_ALWAYS_INLINE s8x64 load(const int8_t* ptr, size_t n, undef<64>) {
  return s8x64{_mm512_maskz_loadu_epi8(internal::mask_x64(n), ptr)};
}

YNN_ALWAYS_INLINE void store(double* ptr, f64x8 val, size_t n) {
  _mm512_mask_storeu_pd(ptr, internal::mask_x8(n), val.v);
}
YNN_ALWAYS_INLINE void store(float* ptr, f32x16 val, size_t n) {
  _mm512_mask_storeu_ps(ptr, internal::mask_x16(n), val.v);
}
YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x16 val, size_t n) {
  _mm512_mask_storeu_epi32(ptr, internal::mask_x16(n), val.v);
}
YNN_ALWAYS_INLINE void store(bfloat16* ptr, bf16x32 val, size_t n) {
  _mm512_mask_storeu_epi16(ptr, internal::mask_x32(n), val.v);
}
YNN_ALWAYS_INLINE void store(half* ptr, f16x32 val, size_t n) {
  _mm512_mask_storeu_epi16(ptr, internal::mask_x32(n), val.v);
}
YNN_ALWAYS_INLINE void store(int16_t* ptr, s16x32 val, size_t n) {
  _mm512_mask_storeu_epi16(ptr, internal::mask_x32(n), val.v);
}
YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x64 val, size_t n) {
  _mm512_mask_storeu_epi8(ptr, internal::mask_x64(n), val.v);
}
YNN_ALWAYS_INLINE void store(int8_t* ptr, s8x64 val, size_t n) {
  _mm512_mask_storeu_epi8(ptr, internal::mask_x64(n), val.v);
}

YNN_ALWAYS_INLINE f64x4 load(const double* ptr, size_t n, f64x4 src) {
  return f64x4{_mm256_mask_loadu_pd(src.v, internal::mask_x4(n), ptr)};
}
YNN_ALWAYS_INLINE f32x8 load(const float* ptr, size_t n, f32x8 src) {
  return f32x8{_mm256_mask_loadu_ps(src.v, internal::mask_x8(n), ptr)};
}
YNN_ALWAYS_INLINE s32x8 load(const int32_t* ptr, size_t n, s32x8 src) {
  return s32x8{_mm256_mask_loadu_epi32(src.v, internal::mask_x8(n), ptr)};
}
YNN_ALWAYS_INLINE bf16x16 load(const bfloat16* ptr, size_t n, bf16x16 src) {
  return bf16x16{_mm256_mask_loadu_epi16(src.v, internal::mask_x16(n), ptr)};
}
YNN_ALWAYS_INLINE f16x16 load(const half* ptr, size_t n, f16x16 src) {
  return f16x16{_mm256_mask_loadu_epi16(src.v, internal::mask_x16(n), ptr)};
}
YNN_ALWAYS_INLINE s16x16 load(const int16_t* ptr, size_t n, s16x16 src) {
  return s16x16{_mm256_mask_loadu_epi16(src.v, internal::mask_x16(n), ptr)};
}
YNN_ALWAYS_INLINE u8x32 load(const uint8_t* ptr, size_t n, u8x32 src) {
  return u8x32{_mm256_mask_loadu_epi8(src.v, internal::mask_x32(n), ptr)};
}
YNN_ALWAYS_INLINE s8x32 load(const int8_t* ptr, size_t n, s8x32 src) {
  return s8x32{_mm256_mask_loadu_epi8(src.v, internal::mask_x32(n), ptr)};
}

YNN_ALWAYS_INLINE f64x4 load(const double* ptr, size_t n, zeros<4>) {
  return f64x4{_mm256_maskz_loadu_pd(internal::mask_x4(n), ptr)};
}
YNN_ALWAYS_INLINE f32x8 load(const float* ptr, size_t n, zeros<8>) {
  return f32x8{_mm256_maskz_loadu_ps(internal::mask_x8(n), ptr)};
}
YNN_ALWAYS_INLINE s32x8 load(const int32_t* ptr, size_t n, zeros<8>) {
  return s32x8{_mm256_maskz_loadu_epi32(internal::mask_x8(n), ptr)};
}
YNN_ALWAYS_INLINE bf16x16 load(const bfloat16* ptr, size_t n, zeros<16>) {
  return bf16x16{_mm256_maskz_loadu_epi16(internal::mask_x16(n), ptr)};
}
YNN_ALWAYS_INLINE f16x16 load(const half* ptr, size_t n, zeros<16>) {
  return f16x16{_mm256_maskz_loadu_epi16(internal::mask_x16(n), ptr)};
}
YNN_ALWAYS_INLINE s16x16 load(const int16_t* ptr, size_t n, zeros<16>) {
  return s16x16{_mm256_maskz_loadu_epi16(internal::mask_x16(n), ptr)};
}
YNN_ALWAYS_INLINE u8x32 load(const uint8_t* ptr, size_t n, zeros<32>) {
  return u8x32{_mm256_maskz_loadu_epi8(internal::mask_x32(n), ptr)};
}
YNN_ALWAYS_INLINE s8x32 load(const int8_t* ptr, size_t n, zeros<32>) {
  return s8x32{_mm256_maskz_loadu_epi8(internal::mask_x32(n), ptr)};
}

YNN_ALWAYS_INLINE f64x4 load(const double* ptr, size_t n, undef<4>) {
  return f64x4{_mm256_maskz_loadu_pd(internal::mask_x4(n), ptr)};
}
YNN_ALWAYS_INLINE f32x8 load(const float* ptr, size_t n, undef<8>) {
  return f32x8{_mm256_maskz_loadu_ps(internal::mask_x8(n), ptr)};
}
YNN_ALWAYS_INLINE s32x8 load(const int32_t* ptr, size_t n, undef<8>) {
  return s32x8{_mm256_maskz_loadu_epi32(internal::mask_x8(n), ptr)};
}
YNN_ALWAYS_INLINE bf16x16 load(const bfloat16* ptr, size_t n, undef<16>) {
  return bf16x16{_mm256_maskz_loadu_epi16(internal::mask_x16(n), ptr)};
}
YNN_ALWAYS_INLINE f16x16 load(const half* ptr, size_t n, undef<16>) {
  return f16x16{_mm256_maskz_loadu_epi16(internal::mask_x16(n), ptr)};
}
YNN_ALWAYS_INLINE s16x16 load(const int16_t* ptr, size_t n, undef<16>) {
  return s16x16{_mm256_maskz_loadu_epi16(internal::mask_x16(n), ptr)};
}
YNN_ALWAYS_INLINE u8x32 load(const uint8_t* ptr, size_t n, undef<32>) {
  return u8x32{_mm256_maskz_loadu_epi8(internal::mask_x32(n), ptr)};
}
YNN_ALWAYS_INLINE s8x32 load(const int8_t* ptr, size_t n, undef<32>) {
  return s8x32{_mm256_maskz_loadu_epi8(internal::mask_x32(n), ptr)};
}

YNN_ALWAYS_INLINE void store(double* ptr, f64x4 val, size_t n) {
  _mm256_mask_storeu_pd(ptr, internal::mask_x4(n), val.v);
}
YNN_ALWAYS_INLINE void store(float* ptr, f32x8 val, size_t n) {
  _mm256_mask_storeu_ps(ptr, internal::mask_x8(n), val.v);
}
YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x8 val, size_t n) {
  _mm256_mask_storeu_epi32(ptr, internal::mask_x8(n), val.v);
}
YNN_ALWAYS_INLINE void store(bfloat16* ptr, bf16x16 val, size_t n) {
  _mm256_mask_storeu_epi16(ptr, internal::mask_x16(n), val.v);
}
YNN_ALWAYS_INLINE void store(half* ptr, f16x16 val, size_t n) {
  _mm256_mask_storeu_epi16(ptr, internal::mask_x16(n), val.v);
}
YNN_ALWAYS_INLINE void store(int16_t* ptr, s16x16 val, size_t n) {
  _mm256_mask_storeu_epi16(ptr, internal::mask_x16(n), val.v);
}
YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x32 val, size_t n) {
  _mm256_mask_storeu_epi8(ptr, internal::mask_x32(n), val.v);
}
YNN_ALWAYS_INLINE void store(int8_t* ptr, s8x32 val, size_t n) {
  _mm256_mask_storeu_epi8(ptr, internal::mask_x32(n), val.v);
}

YNN_ALWAYS_INLINE f64x2 load(const double* ptr, size_t n, f64x2 src) {
  return f64x2{_mm_mask_loadu_pd(src.v, internal::mask_x2(n), ptr)};
}
YNN_ALWAYS_INLINE f32x4 load(const float* ptr, size_t n, f32x4 src) {
  return f32x4{_mm_mask_loadu_ps(src.v, internal::mask_x4(n), ptr)};
}
YNN_ALWAYS_INLINE s32x4 load(const int32_t* ptr, size_t n, s32x4 src) {
  return s32x4{_mm_mask_loadu_epi32(src.v, internal::mask_x4(n), ptr)};
}
YNN_ALWAYS_INLINE bf16x8 load(const bfloat16* ptr, size_t n, bf16x8 src) {
  return bf16x8{_mm_mask_loadu_epi16(src.v, internal::mask_x8(n), ptr)};
}
YNN_ALWAYS_INLINE f16x8 load(const half* ptr, size_t n, f16x8 src) {
  return f16x8{_mm_mask_loadu_epi16(src.v, internal::mask_x8(n), ptr)};
}
YNN_ALWAYS_INLINE s16x8 load(const int16_t* ptr, size_t n, s16x8 src) {
  return s16x8{_mm_mask_loadu_epi16(src.v, internal::mask_x8(n), ptr)};
}
YNN_ALWAYS_INLINE u8x16 load(const uint8_t* ptr, size_t n, u8x16 src) {
  return u8x16{_mm_mask_loadu_epi8(src.v, internal::mask_x16(n), ptr)};
}
YNN_ALWAYS_INLINE s8x16 load(const int8_t* ptr, size_t n, s8x16 src) {
  return s8x16{_mm_mask_loadu_epi8(src.v, internal::mask_x16(n), ptr)};
}

YNN_ALWAYS_INLINE f64x2 load(const double* ptr, size_t n, zeros<2>) {
  return f64x2{_mm_maskz_loadu_pd(internal::mask_x2(n), ptr)};
}
YNN_ALWAYS_INLINE f32x4 load(const float* ptr, size_t n, zeros<4>) {
  return f32x4{_mm_maskz_loadu_ps(internal::mask_x4(n), ptr)};
}
YNN_ALWAYS_INLINE s32x4 load(const int32_t* ptr, size_t n, zeros<4>) {
  return s32x4{_mm_maskz_loadu_epi32(internal::mask_x4(n), ptr)};
}
YNN_ALWAYS_INLINE bf16x8 load(const bfloat16* ptr, size_t n, zeros<8>) {
  return bf16x8{_mm_maskz_loadu_epi16(internal::mask_x8(n), ptr)};
}
YNN_ALWAYS_INLINE f16x8 load(const half* ptr, size_t n, zeros<8>) {
  return f16x8{_mm_maskz_loadu_epi16(internal::mask_x8(n), ptr)};
}
YNN_ALWAYS_INLINE s16x8 load(const int16_t* ptr, size_t n, zeros<8>) {
  return s16x8{_mm_maskz_loadu_epi16(internal::mask_x8(n), ptr)};
}
YNN_ALWAYS_INLINE u8x16 load(const uint8_t* ptr, size_t n, zeros<16>) {
  return u8x16{_mm_maskz_loadu_epi8(internal::mask_x16(n), ptr)};
}
YNN_ALWAYS_INLINE s8x16 load(const int8_t* ptr, size_t n, zeros<16>) {
  return s8x16{_mm_maskz_loadu_epi8(internal::mask_x16(n), ptr)};
}

YNN_ALWAYS_INLINE f64x2 load(const double* ptr, size_t n, undef<2>) {
  return f64x2{_mm_maskz_loadu_pd(internal::mask_x2(n), ptr)};
}
YNN_ALWAYS_INLINE f32x4 load(const float* ptr, size_t n, undef<4>) {
  return f32x4{_mm_maskz_loadu_ps(internal::mask_x4(n), ptr)};
}
YNN_ALWAYS_INLINE s32x4 load(const int32_t* ptr, size_t n, undef<4>) {
  return s32x4{_mm_maskz_loadu_epi32(internal::mask_x4(n), ptr)};
}
YNN_ALWAYS_INLINE bf16x8 load(const bfloat16* ptr, size_t n, undef<8>) {
  return bf16x8{_mm_maskz_loadu_epi16(internal::mask_x8(n), ptr)};
}
YNN_ALWAYS_INLINE f16x8 load(const half* ptr, size_t n, undef<8>) {
  return f16x8{_mm_maskz_loadu_epi16(internal::mask_x8(n), ptr)};
}
YNN_ALWAYS_INLINE s16x8 load(const int16_t* ptr, size_t n, undef<8>) {
  return s16x8{_mm_maskz_loadu_epi16(internal::mask_x8(n), ptr)};
}
YNN_ALWAYS_INLINE u8x16 load(const uint8_t* ptr, size_t n, undef<16>) {
  return u8x16{_mm_maskz_loadu_epi8(internal::mask_x16(n), ptr)};
}
YNN_ALWAYS_INLINE s8x16 load(const int8_t* ptr, size_t n, undef<16>) {
  return s8x16{_mm_maskz_loadu_epi8(internal::mask_x16(n), ptr)};
}

YNN_ALWAYS_INLINE void store(double* ptr, f64x2 val, size_t n) {
  _mm_mask_storeu_pd(ptr, internal::mask_x2(n), val.v);
}
YNN_ALWAYS_INLINE void store(float* ptr, f32x4 val, size_t n) {
  _mm_mask_storeu_ps(ptr, internal::mask_x4(n), val.v);
}
YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x4 val, size_t n) {
  _mm_mask_storeu_epi32(ptr, internal::mask_x4(n), val.v);
}
YNN_ALWAYS_INLINE void store(bfloat16* ptr, bf16x8 val, size_t n) {
  _mm_mask_storeu_epi16(ptr, internal::mask_x8(n), val.v);
}
YNN_ALWAYS_INLINE void store(half* ptr, f16x8 val, size_t n) {
  _mm_mask_storeu_epi16(ptr, internal::mask_x8(n), val.v);
}
YNN_ALWAYS_INLINE void store(int16_t* ptr, s16x8 val, size_t n) {
  _mm_mask_storeu_epi16(ptr, internal::mask_x8(n), val.v);
}
YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x16 val, size_t n) {
  _mm_mask_storeu_epi8(ptr, internal::mask_x16(n), val.v);
}
YNN_ALWAYS_INLINE void store(int8_t* ptr, s8x16 val, size_t n) {
  _mm_mask_storeu_epi8(ptr, internal::mask_x16(n), val.v);
}

YNN_ALWAYS_INLINE f64x8 operator+(f64x8 a, f64x8 b) {
  return f64x8{_mm512_add_pd(a.v, b.v)};
}
YNN_ALWAYS_INLINE f32x16 operator+(f32x16 a, f32x16 b) {
  return f32x16{_mm512_add_ps(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x16 operator+(s32x16 a, s32x16 b) {
  return s32x16{_mm512_add_epi32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x64 operator+(s8x64 a, s8x64 b) {
  return s8x64{_mm512_add_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x64 operator+(u8x64 a, u8x64 b) {
  return u8x64{_mm512_add_epi8(a.v, b.v)};
}

YNN_ALWAYS_INLINE f64x8 operator-(f64x8 a, f64x8 b) {
  return f64x8{_mm512_sub_pd(a.v, b.v)};
}
YNN_ALWAYS_INLINE f32x16 operator-(f32x16 a, f32x16 b) {
  return f32x16{_mm512_sub_ps(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x16 operator-(s32x16 a, s32x16 b) {
  return s32x16{_mm512_sub_epi32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x64 operator-(s8x64 a, s8x64 b) {
  return s8x64{_mm512_sub_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x64 operator-(u8x64 a, u8x64 b) {
  return u8x64{_mm512_sub_epi8(a.v, b.v)};
}

YNN_ALWAYS_INLINE s16x32 add_sat(s16x32 a, s16x32 b) {
  return s16x32{_mm512_adds_epi16(a.v, b.v)};
}
YNN_ALWAYS_INLINE u16x32 add_sat(u16x32 a, u16x32 b) {
  return u16x32{_mm512_adds_epu16(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x64 add_sat(s8x64 a, s8x64 b) {
  return s8x64{_mm512_adds_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x64 add_sat(u8x64 a, u8x64 b) {
  return u8x64{_mm512_adds_epu8(a.v, b.v)};
}

YNN_ALWAYS_INLINE s16x32 sub_sat(s16x32 a, s16x32 b) {
  return s16x32{_mm512_subs_epi16(a.v, b.v)};
}
YNN_ALWAYS_INLINE u16x32 sub_sat(u16x32 a, u16x32 b) {
  return u16x32{_mm512_subs_epu16(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x64 sub_sat(s8x64 a, s8x64 b) {
  return s8x64{_mm512_subs_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x64 sub_sat(u8x64 a, u8x64 b) {
  return u8x64{_mm512_subs_epu8(a.v, b.v)};
}

YNN_ALWAYS_INLINE f64x8 operator*(f64x8 a, f64x8 b) {
  return f64x8{_mm512_mul_pd(a.v, b.v)};
}
YNN_ALWAYS_INLINE f32x16 operator*(f32x16 a, f32x16 b) {
  return f32x16{_mm512_mul_ps(a.v, b.v)};
}
YNN_ALWAYS_INLINE f64x8 operator/(f64x8 a, f64x8 b) {
  return f64x8{_mm512_div_pd(a.v, b.v)};
}
YNN_ALWAYS_INLINE f32x16 operator/(f32x16 a, f32x16 b) {
  return f32x16{_mm512_div_ps(a.v, b.v)};
}

YNN_ALWAYS_INLINE s32x16 operator*(s32x16 a, s32x16 b) {
  return s32x16{_mm512_mullo_epi32(a.v, b.v)};
}

YNN_ALWAYS_INLINE s16x32 operator&(s16x32 a, s16x32 b) {
  return s16x32{_mm512_and_si512(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x32 operator|(s16x32 a, s16x32 b) {
  return s16x32{_mm512_or_si512(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x32 operator^(s16x32 a, s16x32 b) {
  return s16x32{_mm512_xor_si512(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x32 operator~(s16x32 a) {
  return s16x32{_mm512_xor_si512(a.v, _mm512_set1_epi32(-1))};
}
YNN_ALWAYS_INLINE s16x32 operator&(s16x32 a, int b) {
  return s16x32{_mm512_and_si512(a.v, _mm512_set1_epi16(b))};
}
YNN_ALWAYS_INLINE s16x32 operator>>(s16x32 a, int b) {
  return s16x32{_mm512_srai_epi16(a.v, b)};
}
YNN_ALWAYS_INLINE s16x32 operator<<(s16x32 a, int b) {
  return s16x32{_mm512_slli_epi16(a.v, b)};
}
YNN_ALWAYS_INLINE s32x16 operator&(s32x16 a, s32x16 b) {
  return s32x16{_mm512_and_si512(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x16 operator|(s32x16 a, s32x16 b) {
  return s32x16{_mm512_or_si512(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x16 operator^(s32x16 a, s32x16 b) {
  return s32x16{_mm512_xor_si512(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x16 operator~(s32x16 a) {
  return s32x16{_mm512_xor_si512(a.v, _mm512_set1_epi32(-1))};
}
YNN_ALWAYS_INLINE s32x16 operator<<(s32x16 a, int b) {
  return s32x16{_mm512_slli_epi32(a.v, b)};
}

YNN_ALWAYS_INLINE u8x64 operator&(u8x64 a, u8x64 b) {
  return u8x64{_mm512_and_si512(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x64 operator|(u8x64 a, u8x64 b) {
  return u8x64{_mm512_or_si512(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x64 operator^(u8x64 a, u8x64 b) {
  return u8x64{_mm512_xor_si512(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x64 operator~(u8x64 a) {
  return u8x64{_mm512_xor_si512(a.v, _mm512_set1_epi32(-1))};
}

YNN_ALWAYS_INLINE s8x64 operator&(s8x64 a, s8x64 b) {
  return s8x64{_mm512_and_si512(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x64 operator|(s8x64 a, s8x64 b) {
  return s8x64{_mm512_or_si512(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x64 operator^(s8x64 a, s8x64 b) {
  return s8x64{_mm512_xor_si512(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x64 operator~(s8x64 a) {
  return s8x64{_mm512_xor_si512(a.v, _mm512_set1_epi32(-1))};
}

YNN_ALWAYS_INLINE f64x8 fma(f64x8 a, f64x8 b, f64x8 acc) {
  return f64x8{_mm512_fmadd_pd(a.v, b.v, acc.v)};
}

YNN_ALWAYS_INLINE f32x16 fma(f32x16 a, f32x16 b, f32x16 acc) {
  return f32x16{_mm512_fmadd_ps(a.v, b.v, acc.v)};
}

YNN_ALWAYS_INLINE f64x8 min(f64x8 a, f64x8 b) {
  return f64x8{_mm512_min_pd(a.v, b.v)};
}
YNN_ALWAYS_INLINE f32x16 min(f32x16 a, f32x16 b) {
  return f32x16{_mm512_min_ps(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x16 min(s32x16 a, s32x16 b) {
  return s32x16{_mm512_min_epi32(a.v, b.v)};
}

YNN_ALWAYS_INLINE f64x8 max(f64x8 a, f64x8 b) {
  return f64x8{_mm512_max_pd(a.v, b.v)};
}
YNN_ALWAYS_INLINE f32x16 max(f32x16 a, f32x16 b) {
  return f32x16{_mm512_max_ps(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x16 max(s32x16 a, s32x16 b) {
  return s32x16{_mm512_max_epi32(a.v, b.v)};
}
YNN_ALWAYS_INLINE f64x8 floor(f64x8 a) { return f64x8{_mm512_floor_pd(a.v)}; }
YNN_ALWAYS_INLINE f64x8 ceil(f64x8 a) { return f64x8{_mm512_ceil_pd(a.v)}; }
YNN_ALWAYS_INLINE f64x8 round(f64x8 a) {
  return f64x8{
      _mm512_roundscale_pd(a.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)};
}
YNN_ALWAYS_INLINE f64x8 sqrt(f64x8 a) { return f64x8{_mm512_sqrt_pd(a.v)}; }
YNN_ALWAYS_INLINE f32x16 floor(f32x16 a) {
  return f32x16{_mm512_floor_ps(a.v)};
}
YNN_ALWAYS_INLINE f32x16 ceil(f32x16 a) { return f32x16{_mm512_ceil_ps(a.v)}; }
YNN_ALWAYS_INLINE f32x16 round(f32x16 a) {
  return f32x16{
      _mm512_roundscale_ps(a.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)};
}
YNN_ALWAYS_INLINE f32x16 sqrt(f32x16 a) { return f32x16{_mm512_sqrt_ps(a.v)}; }

YNN_ALWAYS_INLINE s16x32 min(s16x32 a, s16x32 b) {
  return s16x32{_mm512_min_epi16(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x64 min(s8x64 a, s8x64 b) {
  return s8x64{_mm512_min_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x64 min(u8x64 a, u8x64 b) {
  return u8x64{_mm512_min_epu8(a.v, b.v)};
}

YNN_ALWAYS_INLINE s16x32 max(s16x32 a, s16x32 b) {
  return s16x32{_mm512_max_epi16(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x64 max(s8x64 a, s8x64 b) {
  return s8x64{_mm512_max_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x64 max(u8x64 a, u8x64 b) {
  return u8x64{_mm512_max_epu8(a.v, b.v)};
}

YNN_ALWAYS_INLINE u8x64 abs(s8x64 a) { return u8x64{_mm512_abs_epi8(a.v)}; }
YNN_ALWAYS_INLINE u16x32 abs(s16x32 a) { return u16x32{_mm512_abs_epi16(a.v)}; }
YNN_ALWAYS_INLINE u32x16 abs(s32x16 a) { return u32x16{_mm512_abs_epi32(a.v)}; }
YNN_ALWAYS_INLINE f64x8 abs(f64x8 a) { return f64x8{_mm512_abs_pd(a.v)}; }
YNN_ALWAYS_INLINE f32x16 abs(f32x16 a) { return f32x16{_mm512_abs_ps(a.v)}; }

YNN_ALWAYS_INLINE f32x4 floor_log2(f32x4 a) {
  // getexp handles 0 correctly, but not negative numbers.
  __m128 res = _mm_getexp_ps(a.v);
  __mmask8 negative = _mm_cmp_ps_mask(a.v, _mm_setzero_ps(), _CMP_LT_OQ);
  return f32x4{_mm_mask_blend_ps(
      negative, res, _mm_set1_ps(std::numeric_limits<float>::quiet_NaN()))};
}
YNN_ALWAYS_INLINE f32x8 floor_log2(f32x8 a) {
  // getexp handles 0 correctly, but not negative numbers.
  __m256 res = _mm256_getexp_ps(a.v);
  __mmask8 negative = _mm256_cmp_ps_mask(a.v, _mm256_setzero_ps(), _CMP_LT_OQ);
  return f32x8{_mm256_mask_blend_ps(
      negative, res, _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN()))};
}
YNN_ALWAYS_INLINE f32x16 floor_log2(f32x16 a) {
  // getexp handles 0 correctly, but not negative numbers.
  __m512 res = _mm512_getexp_ps(a.v);
  __mmask16 negative = _mm512_cmp_ps_mask(a.v, _mm512_setzero_ps(), _CMP_LT_OQ);
  return f32x16{_mm512_mask_blend_ps(
      negative, res, _mm512_set1_ps(std::numeric_limits<float>::quiet_NaN()))};
}
YNN_ALWAYS_INLINE f64x2 floor_log2(f64x2 a) {
  // getexp handles 0 correctly, but not negative numbers.
  __m128d res = _mm_getexp_pd(a.v);
  __mmask8 negative = _mm_cmp_pd_mask(a.v, _mm_setzero_pd(), _CMP_LT_OQ);
  return f64x2{_mm_mask_blend_pd(
      negative, res, _mm_set1_pd(std::numeric_limits<double>::quiet_NaN()))};
}
YNN_ALWAYS_INLINE f64x4 floor_log2(f64x4 a) {
  // getexp handles 0 correctly, but not negative numbers.
  __m256d res = _mm256_getexp_pd(a.v);
  __mmask8 negative = _mm256_cmp_pd_mask(a.v, _mm256_setzero_pd(), _CMP_LT_OQ);
  return f64x4{_mm256_mask_blend_pd(
      negative, res, _mm256_set1_pd(std::numeric_limits<double>::quiet_NaN()))};
}
YNN_ALWAYS_INLINE f64x8 floor_log2(f64x8 a) {
  // getexp handles 0 correctly, but not negative numbers.
  __m512d res = _mm512_getexp_pd(a.v);
  __mmask8 negative = _mm512_cmp_pd_mask(a.v, _mm512_setzero_pd(), _CMP_LT_OQ);
  return f64x8{_mm512_mask_blend_pd(
      negative, res, _mm512_set1_pd(std::numeric_limits<double>::quiet_NaN()))};
}

YNN_ALWAYS_INLINE f32x16 exp2_round(f32x16 a) {
  const __m512 magic = _mm512_set1_ps(127.0f + static_cast<float>(1 << 23));
  const __m512 res_bits = _mm512_add_ps(a.v, magic);
  return f32x16{_mm512_castsi512_ps(
      _mm512_slli_epi32(_mm512_castps_si512(res_bits), 23))};
}
YNN_ALWAYS_INLINE f64x8 exp2_round(f64x8 a) {
  const __m512d magic = _mm512_set1_pd(1023.0 + static_cast<double>(1ll << 52));
  const __m512d res_bits = _mm512_add_pd(a.v, magic);
  return f64x8{_mm512_castsi512_pd(
      _mm512_slli_epi64(_mm512_castpd_si512(res_bits), 52))};
}

YNN_ALWAYS_INLINE f32x16 copynan(f32x16 x, f32x16 nan) {
  __mmask16 is_nan = _mm512_fpclass_ps_mask(nan.v, 0x81);
  return f32x16{_mm512_mask_mov_ps(x.v, is_nan, nan.v)};
}
YNN_ALWAYS_INLINE f64x8 copynan(f64x8 x, f64x8 nan) {
  __mmask8 is_nan = _mm512_fpclass_pd_mask(nan.v, 0x81);
  return f64x8{_mm512_mask_mov_pd(x.v, is_nan, nan.v)};
}

YNN_ALWAYS_INLINE void kahan_sum(f32x16 a, f32x16& acc, f32x16& error) {
  f32x16 y = a - error;
  f32x16 t = acc + y;
  error = (t - acc) - y;
  __m512 mask = _mm512_set1_ps(std::numeric_limits<float>::infinity());
  __mmask16 m =
      _mm512_cmp_ps_mask(_mm512_and_ps(error.v, mask), mask, _CMP_NEQ_OQ);
  error = f32x16{_mm512_maskz_mov_ps(m, error.v)};
  acc = t;
}

YNN_ALWAYS_INLINE void kahan_sum(f64x8 a, f64x8& acc, f64x8& error) {
  f64x8 y = a - error;
  f64x8 t = acc + y;
  error = (t - acc) - y;
  __m512d mask = _mm512_set1_pd(std::numeric_limits<double>::infinity());
  __mmask8 m =
      _mm512_cmp_pd_mask(_mm512_and_pd(error.v, mask), mask, _CMP_NEQ_OQ);
  error = f64x8{_mm512_maskz_mov_pd(m, error.v)};
  acc = t;
}

template <int Index>
YNN_ALWAYS_INLINE s32x4 extract(s32x16 x, decltype(s32x4::N)) {
  return s32x4{_mm512_extracti32x4_epi32(x.v, Index)};
}
template <int Index>
YNN_ALWAYS_INLINE f32x4 extract(f32x16 x, decltype(f32x4::N)) {
  return f32x4{_mm512_extractf32x4_ps(x.v, Index)};
}
template <int Index>
YNN_ALWAYS_INLINE s8x16 extract(s8x64 x, decltype(s8x16::N)) {
  return s8x16{_mm512_extracti32x4_epi32(x.v, Index)};
}
template <int Index>
YNN_ALWAYS_INLINE u8x16 extract(u8x64 x, decltype(u8x16::N)) {
  return u8x16{_mm512_extracti32x4_epi32(x.v, Index)};
}

YNN_ALWAYS_INLINE std::tuple<u8x64, u8x64> interleave(
    std::integral_constant<size_t, 256>, u8x64 x0, u8x64 x1) {
  return {u8x64{_mm512_shuffle_i64x2(x0.v, x1.v, 0x44)},
          u8x64{_mm512_shuffle_i64x2(x0.v, x1.v, 0xEE)}};
}
YNN_ALWAYS_INLINE std::tuple<u8x64, u8x64> interleave(
    std::integral_constant<size_t, 128>, u8x64 x0, u8x64 x1) {
  const __m512i idx1 = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
  const __m512i idx2 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
  return {u8x64{_mm512_permutex2var_epi64(x0.v, idx1, x1.v)},
          u8x64{_mm512_permutex2var_epi64(x0.v, idx2, x1.v)}};
}
YNN_ALWAYS_INLINE std::tuple<u8x64, u8x64> interleave(
    std::integral_constant<size_t, 64>, u8x64 x0, u8x64 x1) {
  const __m512i idx1 = _mm512_set_epi64(11, 3, 10, 2, 9, 1, 8, 0);
  const __m512i idx2 = _mm512_set_epi64(15, 7, 14, 6, 13, 5, 12, 4);
  return {u8x64{_mm512_permutex2var_epi64(x0.v, idx1, x1.v)},
          u8x64{_mm512_permutex2var_epi64(x0.v, idx2, x1.v)}};
}
YNN_ALWAYS_INLINE std::tuple<u8x64, u8x64> interleave(
    std::integral_constant<size_t, 32>, u8x64 x0, u8x64 x1) {
  const __m512i idx1 =
      _mm512_set_epi32(23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
  const __m512i idx2 = _mm512_set_epi32(31, 15, 30, 14, 29, 13, 28, 12, 27, 11,
                                        26, 10, 25, 9, 24, 8);
  return {u8x64{_mm512_permutex2var_epi32(x0.v, idx1, x1.v)},
          u8x64{_mm512_permutex2var_epi32(x0.v, idx2, x1.v)}};
}
YNN_ALWAYS_INLINE std::tuple<u8x64, u8x64> interleave(
    std::integral_constant<size_t, 16>, u8x64 x0, u8x64 x1) {
  return interleave(std::integral_constant<size_t, 128>{},
                    u8x64{_mm512_unpacklo_epi16(x0.v, x1.v)},
                    u8x64{_mm512_unpackhi_epi16(x0.v, x1.v)});
}
YNN_ALWAYS_INLINE std::tuple<u8x64, u8x64> interleave(
    std::integral_constant<size_t, 8>, u8x64 x0, u8x64 x1) {
  return interleave(std::integral_constant<size_t, 128>{},
                    u8x64{_mm512_unpacklo_epi8(x0.v, x1.v)},
                    u8x64{_mm512_unpackhi_epi8(x0.v, x1.v)});
}
YNN_ALWAYS_INLINE std::tuple<u8x64, u8x64> interleave(
    std::integral_constant<size_t, 4>, u8x64 x0, u8x64 x1) {
  __m512i even0 = _mm512_and_si512(x0.v, _mm512_set1_epi8(0x0f));
  __m512i even1 = _mm512_and_si512(x1.v, _mm512_set1_epi8(0x0f));
  __m512i odd0 = _mm512_and_si512(x0.v, _mm512_set1_epi8(0xf0));
  __m512i odd1 = _mm512_and_si512(x1.v, _mm512_set1_epi8(0xf0));
  return interleave(std::integral_constant<size_t, 8>{},
                    u8x64{_mm512_or_si512(_mm512_slli_epi16(even1, 4), even0)},
                    u8x64{_mm512_or_si512(odd1, _mm512_srli_epi16(odd0, 4))});
}
YNN_ALWAYS_INLINE std::tuple<u8x64, u8x64> interleave(
    std::integral_constant<size_t, 2>, u8x64 x0, u8x64 x1) {
  __m512i even0 = _mm512_and_si512(x0.v, _mm512_set1_epi8(0x33));
  __m512i even1 = _mm512_and_si512(x1.v, _mm512_set1_epi8(0x33));
  __m512i odd0 = _mm512_and_si512(x0.v, _mm512_set1_epi8(0xcc));
  __m512i odd1 = _mm512_and_si512(x1.v, _mm512_set1_epi8(0xcc));
  return interleave(std::integral_constant<size_t, 4>{},
                    u8x64{_mm512_or_si512(_mm512_slli_epi16(even1, 2), even0)},
                    u8x64{_mm512_or_si512(odd1, _mm512_srli_epi16(odd0, 2))});
}

using f32x32 = vec<float, 32>;
using s16x64 = vec<int16_t, 64>;
using s32x32 = vec<int32_t, 32>;
using s32x64 = vec<int32_t, 64>;

YNN_ALWAYS_INLINE f32x16 cast(f16x16 x, float) {
  return f32x16{_mm512_cvtph_ps(x.v)};
}

YNN_ALWAYS_INLINE f32x16 cast(bf16x16 a, float) {
  return f32x16{_mm512_castsi512_ps(_mm512_slli_epi32(
      _mm512_cvtepu16_epi32(a.v), 16))};
}

YNN_ALWAYS_INLINE bf16x32 cast(f32x32 a, bfloat16) {
#ifdef YNN_ARCH_X86_AVX512BF16
  return bf16x32{(__m512i)_mm512_cvtne2ps_pbh(hi(a).v, lo(a).v)};
#else
  __m512i u_lo = _mm512_castps_si512(lo(a).v);
  __mmask16 nan_mask_lo = _mm512_cmp_ps_mask(lo(a).v, lo(a).v, _CMP_UNORD_Q);
  __m512i lsb_lo =
      _mm512_and_si512(_mm512_srli_epi32(u_lo, 16), _mm512_set1_epi32(1));
  __m512i bias_lo = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb_lo);
  __m512i res_lo =
      _mm512_mask_or_epi32(_mm512_add_epi32(u_lo, bias_lo), nan_mask_lo, u_lo,
                           _mm512_set1_epi32(0x00010000));
  __m512i c1 = _mm512_srli_epi32(res_lo, 16);

  __m512i u_hi = _mm512_castps_si512(hi(a).v);
  __mmask16 nan_mask_hi = _mm512_cmp_ps_mask(hi(a).v, hi(a).v, _CMP_UNORD_Q);
  __m512i lsb_hi =
      _mm512_and_si512(_mm512_srli_epi32(u_hi, 16), _mm512_set1_epi32(1));
  __m512i bias_hi = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb_hi);
  __m512i res_hi =
      _mm512_mask_or_epi32(_mm512_add_epi32(u_hi, bias_hi), nan_mask_hi, u_hi,
                           _mm512_set1_epi32(0x00010000));
  __m512i c2 = _mm512_srli_epi32(res_hi, 16);

  const __m512i d = _mm512_packus_epi32(c1, c2);
  const __m512i permutation =
      _mm512_set_epi32(15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0);
  return bf16x32{_mm512_permutevar_epi32(permutation, d)};
#endif
}
YNN_ALWAYS_INLINE bf16x16 cast(f32x16 a, bfloat16) {
#ifdef YNN_ARCH_X86_AVX512BF16
  return bf16x16{
      (__m256i)_mm256_cvtne2ps_pbh(internal::hi(a.v), internal::lo(a.v))};
#else
  __m512i u = _mm512_castps_si512(a.v);
  __mmask16 nan_mask = _mm512_cmp_ps_mask(a.v, a.v, _CMP_UNORD_Q);
  __m512i lsb =
      _mm512_and_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1));
  __m512i bias = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb);
  __m512i res = _mm512_mask_or_epi32(_mm512_add_epi32(u, bias), nan_mask, u,
                                     _mm512_set1_epi32(0x00010000));
  __m512i c = _mm512_srli_epi32(res, 16);
  const __m256i d = _mm256_packus_epi32(internal::lo(c), internal::hi(c));
  return bf16x16{_mm256_permute4x64_epi64(d, _MM_SHUFFLE(3, 1, 2, 0))};
#endif
}

YNN_ALWAYS_INLINE s32x16 cast(s8x16 a, int32_t) {
  return s32x16{_mm512_cvtepi8_epi32(a.v)};
}
YNN_ALWAYS_INLINE s32x16 cast(u8x16 a, int32_t) {
  return s32x16{_mm512_cvtepu8_epi32(a.v)};
}

YNN_ALWAYS_INLINE s16x32 cast(s8x32 a, int16_t) {
  return s16x32{_mm512_cvtepi8_epi16(a.v)};
}
YNN_ALWAYS_INLINE s16x32 cast(u8x32 a, int16_t) {
  return s16x32{_mm512_cvtepu8_epi16(a.v)};
}

YNN_ALWAYS_INLINE f32x16 cast(s32x16 x, float) {
  return f32x16{_mm512_cvtepi32_ps(x.v)};
}

YNN_ALWAYS_INLINE f64x8 cast(f32x8 a, double) {
  return f64x8{_mm512_cvtps_pd(a.v)};
}
YNN_ALWAYS_INLINE f32x8 cast(f64x8 a, float) {
  return f32x8{_mm512_cvtpd_ps(a.v)};
}

YNN_ALWAYS_INLINE s16x32 cast(s32x32 a, int16_t) {
  const __m512i r = _mm512_packs_epi32(lo(a).v, hi(a).v);
  return s16x32{
      _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), r)};
}

YNN_ALWAYS_INLINE s8x64 cast(s16x64 a, int8_t) {
  const __m512i r = _mm512_packs_epi16(lo(a).v, hi(a).v);
  return s8x64{
      _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), r)};
}

YNN_ALWAYS_INLINE u8x64 cast(s16x64 a, uint8_t) {
  const __m512i r = _mm512_packus_epi16(lo(a).v, hi(a).v);
  return u8x64{
      _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), r)};
}

YNN_ALWAYS_INLINE s32x16 cast(f32x16 f, int32_t) {
  const __mmask16 mask =
      _mm512_cmp_ps_mask(f.v, _mm512_set1_ps(2147483520.0f), _CMP_GT_OQ);
  const __m512i res = _mm512_cvt_roundps_epi32(
      f.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  return s32x16{
      _mm512_mask_blend_epi32(mask, res, _mm512_set1_epi32(0x7fffffff))};
}

YNN_ALWAYS_INLINE s16x32 cast(f32x32 f, int16_t) {
  const s32x16 i0 = cast(lo(f), int32_t());
  const s32x16 i1 = cast(hi(f), int32_t());
  return cast(s32x32(i0, i1), int16_t());
}

YNN_ALWAYS_INLINE u8x64 cast(f32x64 f, uint8_t) {
  const s32x16 i0 = cast(lo(lo(f)), int32_t());
  const s32x16 i1 = cast(hi(lo(f)), int32_t());
  const s32x16 i2 = cast(lo(hi(f)), int32_t());
  const s32x16 i3 = cast(hi(hi(f)), int32_t());
  const __m512i i01_16 = _mm512_packs_epi32(i0.v, i1.v);
  const __m512i i23_16 = _mm512_packs_epi32(i2.v, i3.v);
  const __m512i r = _mm512_packus_epi16(i01_16, i23_16);
  const __m512i idx =
      _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
  return u8x64{_mm512_permutexvar_epi32(idx, r)};
}

YNN_ALWAYS_INLINE s8x64 cast(f32x64 f, int8_t) {
  const s32x16 i0 = cast(lo(lo(f)), int32_t());
  const s32x16 i1 = cast(hi(lo(f)), int32_t());
  const s32x16 i2 = cast(lo(hi(f)), int32_t());
  const s32x16 i3 = cast(hi(hi(f)), int32_t());
  const __m512i i01_16 = _mm512_packs_epi32(i0.v, i1.v);
  const __m512i i23_16 = _mm512_packs_epi32(i2.v, i3.v);
  const __m512i r = _mm512_packs_epi16(i01_16, i23_16);
  const __m512i idx =
      _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
  return s8x64{_mm512_permutexvar_epi32(idx, r)};
}

YNN_ALWAYS_INLINE s8x64 cast(s2x64 from, int8_t) {
  // 1. Broadcast the 128-bit input across all four 128-bit lanes of YMM.
  __m512i dup = _mm512_broadcast_i32x4(from.v);

  // 2. Duplicate the bytes so each 2-bit value has its own 8-bit lane.
  const __m512i mask_dup = _mm512_set_epi8(
      15, 15, 15, 15, 14, 14, 14, 14, 13, 13, 13, 13, 12, 12, 12, 12, 11, 11,
      11, 11, 10, 10, 10, 10, 9, 9, 9, 9, 8, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 6, 5,
      5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  dup = _mm512_shuffle_epi8(dup, mask_dup);

  // 3. The cross-byte spill trick.
  __m512i shifted = _mm512_srli_epi32(dup, 4);
  __m512i blended = _mm512_mask_blend_epi16(0xAAAAAAAA, dup, shifted);
  __m512i masked = _mm512_and_si512(blended, _mm512_set1_epi32(0x0C030C03));

  // 4. Final sign-extension LUT
  const __m512i lut = _mm512_set_epi8(
      0, 0, 0, -1, 0, 0, 0, -2, 0, 0, 0, 1, -1, -2, 1, 0, 0, 0, 0, -1, 0, 0, 0,
      -2, 0, 0, 0, 1, -1, -2, 1, 0, 0, 0, 0, -1, 0, 0, 0, -2, 0, 0, 0, 1, -1,
      -2, 1, 0, 0, 0, 0, -1, 0, 0, 0, -2, 0, 0, 0, 1, -1, -2, 1, 0);

  return s8x64{_mm512_shuffle_epi8(lut, masked)};
}

YNN_ALWAYS_INLINE s8x64 cast(s4x64 from, int8_t) {
  // 1. Broadcast the 256-bit input to both halves of the 512-bit register.
  __m512i from512 = _mm512_castsi256_si512(from.v);
  __m512i dup = _mm512_shuffle_i64x2(from512, from512, _MM_SHUFFLE(1, 1, 0, 0));

  // 2. Duplicate each byte 2 times inside each 128-bit lane.
  const __m512i mask_dup = _mm512_set_epi8(
      15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5,
      5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 15, 15, 14, 14, 13, 13, 12, 12, 11, 11,
      10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
  dup = _mm512_shuffle_epi8(dup, mask_dup);

  // 3. Shift right and mask-blend bytes
  __mmask64 mask1 = 0xAAAAAAAAAAAAAAAAULL;
  __m512i shifted = _mm512_srli_epi16(dup, 4);
  __m512i blended = _mm512_mask_blend_epi8(mask1, dup, shifted);

  // 4. Mask indices and perform sign-extension LUT
  __m512i indices = _mm512_and_si512(blended, _mm512_set1_epi8(0x0f));
  const __m512i lut =
      _mm512_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0,
                      -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0,
                      -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0,
                      -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0);

  return s8x64{_mm512_shuffle_epi8(lut, indices)};
}
}  // namespace simd

}  // namespace ynn

#include "ynnpack/base/simd/generic.inc"  // IWYU pragma: export

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_AVX512F_H_
