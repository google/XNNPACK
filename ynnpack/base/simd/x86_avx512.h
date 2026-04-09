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

  YNN_ALWAYS_INLINE f64x4 lo() const { return f64x4{internal::lo(v)}; }
  YNN_ALWAYS_INLINE f64x4 hi() const { return f64x4{internal::hi(v)}; }
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

  YNN_ALWAYS_INLINE f32x8 lo() const { return f32x8{internal::lo(v)}; }
  YNN_ALWAYS_INLINE f32x8 hi() const { return f32x8{internal::hi(v)}; }
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

  YNN_ALWAYS_INLINE u32x8 lo() const { return u32x8{internal::lo(v)}; }
  YNN_ALWAYS_INLINE u32x8 hi() const { return u32x8{internal::hi(v)}; }
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

  YNN_ALWAYS_INLINE s32x8 lo() const { return s32x8{internal::lo(v)}; }
  YNN_ALWAYS_INLINE s32x8 hi() const { return s32x8{internal::hi(v)}; }
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

  YNN_ALWAYS_INLINE bf16x16 lo() const { return bf16x16{internal::lo(v)}; }
  YNN_ALWAYS_INLINE bf16x16 hi() const { return bf16x16{internal::hi(v)}; }
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

  YNN_ALWAYS_INLINE f16x16 lo() const { return f16x16{internal::lo(v)}; }
  YNN_ALWAYS_INLINE f16x16 hi() const { return f16x16{internal::hi(v)}; }
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

  YNN_ALWAYS_INLINE u16x16 lo() const { return u16x16{internal::lo(v)}; }
  YNN_ALWAYS_INLINE u16x16 hi() const { return u16x16{internal::hi(v)}; }
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

  YNN_ALWAYS_INLINE s16x16 lo() const { return s16x16{internal::lo(v)}; }
  YNN_ALWAYS_INLINE s16x16 hi() const { return s16x16{internal::hi(v)}; }
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

  YNN_ALWAYS_INLINE u8x32 lo() const { return u8x32{internal::lo(v)}; }
  YNN_ALWAYS_INLINE u8x32 hi() const { return u8x32{internal::hi(v)}; }
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

  YNN_ALWAYS_INLINE s8x32 lo() const { return s8x32{internal::lo(v)}; }
  YNN_ALWAYS_INLINE s8x32 hi() const { return s8x32{internal::hi(v)}; }
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

#ifdef YNN_ARCH_X86_AVX512BF16
YNN_ALWAYS_INLINE bf16x32 cast(f32x32 a, bfloat16) {
  return bf16x32{(__m512i)_mm512_cvtne2ps_pbh(a.hi().v, a.lo().v)};
}
#endif

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

YNN_ALWAYS_INLINE s32x16 cast(f32x16 x, int32_t) {
  return s32x16{_mm512_cvttps_epi32(x.v)};
}

YNN_ALWAYS_INLINE s16x32 saturate_cast(s32x32 a, int16_t) {
  const __m512i r = _mm512_packs_epi32(a.lo().v, a.hi().v);
  return s16x32{
      _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), r)};
}

YNN_ALWAYS_INLINE s8x64 saturate_cast(s16x64 a, int8_t) {
  const __m512i r = _mm512_packs_epi16(a.lo().v, a.hi().v);
  return s8x64{
      _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), r)};
}

YNN_ALWAYS_INLINE u8x64 saturate_cast(s16x64 a, uint8_t) {
  const __m512i r = _mm512_packus_epi16(a.lo().v, a.hi().v);
  return u8x64{
      _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), r)};
}

YNN_ALWAYS_INLINE s16x32 round_float_to_int(f32x32 f, int16_t) {
  const __m512 max_int16 = _mm512_set1_ps((1 << 15) - 1);
  const __m512i i0 = _mm512_cvtps_epi32(_mm512_min_ps(f.lo().v, max_int16));
  const __m512i i1 = _mm512_cvtps_epi32(_mm512_min_ps(f.hi().v, max_int16));
  return saturate_cast(s32x32(s32x16(i0), s32x16(i1)), int16_t());
}

YNN_ALWAYS_INLINE u8x64 round_float_to_int(f32x64 f, uint8_t) {
  const __m512 max_uint16 = _mm512_set1_ps((1 << 16) - 1);
  const __m512i i0 =
      _mm512_cvtps_epi32(_mm512_min_ps(f.lo().lo().v, max_uint16));
  const __m512i i1 =
      _mm512_cvtps_epi32(_mm512_min_ps(f.lo().hi().v, max_uint16));
  const __m512i i2 =
      _mm512_cvtps_epi32(_mm512_min_ps(f.hi().lo().v, max_uint16));
  const __m512i i3 =
      _mm512_cvtps_epi32(_mm512_min_ps(f.hi().hi().v, max_uint16));
  const __m512i i01_16 = _mm512_packs_epi32(i0, i1);
  const __m512i i23_16 = _mm512_packs_epi32(i2, i3);
  const __m512i r = _mm512_packus_epi16(i01_16, i23_16);
  const __m512i idx =
      _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
  return u8x64{_mm512_permutexvar_epi32(idx, r)};
}

YNN_ALWAYS_INLINE s8x64 round_float_to_int(f32x64 f, int8_t) {
  const __m512 max_int16 = _mm512_set1_ps((1 << 15) - 1);
  const __m512i i0 =
      _mm512_cvtps_epi32(_mm512_min_ps(f.lo().lo().v, max_int16));
  const __m512i i1 =
      _mm512_cvtps_epi32(_mm512_min_ps(f.lo().hi().v, max_int16));
  const __m512i i2 =
      _mm512_cvtps_epi32(_mm512_min_ps(f.hi().lo().v, max_int16));
  const __m512i i3 =
      _mm512_cvtps_epi32(_mm512_min_ps(f.hi().hi().v, max_int16));
  const __m512i i01_16 = _mm512_packs_epi32(i0, i1);
  const __m512i i23_16 = _mm512_packs_epi32(i2, i3);
  const __m512i r = _mm512_packs_epi16(i01_16, i23_16);
  const __m512i idx =
      _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
  return s8x64{_mm512_permutexvar_epi32(idx, r)};
}

}  // namespace simd

}  // namespace ynn

#include "ynnpack/base/simd/generic.inc"  // IWYU pragma: export

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_AVX512F_H_
