// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_VEC256_BASE_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_VEC256_BASE_H_

#ifndef YNN_ARCH_X86_AVX
#error "x86_vec256.h requires AVX"
#endif  // YNN_ARCH_X86_AVX

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/bit_cast.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/simd/x86_vec128_base.h"  // IWYU pragma: export

namespace ynn {

namespace simd {

// See vec.h for architecture independent comments.

namespace internal {

YNN_ALWAYS_INLINE __m128 lo(__m256 x) { return _mm256_castps256_ps128(x); }
YNN_ALWAYS_INLINE __m128 hi(__m256 x) { return _mm256_extractf128_ps(x, 1); }

YNN_ALWAYS_INLINE __m128i lo(__m256i x) { return _mm256_castsi256_si128(x); }
YNN_ALWAYS_INLINE __m128i hi(__m256i x) {
  return _mm_castps_si128(_mm256_extractf128_ps(_mm256_castsi256_ps(x), 1));
}

YNN_ALWAYS_INLINE __m128d lo(__m256d x) {return _mm256_castpd256_pd128(x); }
YNN_ALWAYS_INLINE __m128d hi(__m256d x) { return _mm256_extractf128_pd(x, 1); }

YNN_ALWAYS_INLINE __m256 concat(__m128 lo, __m128 hi) {
  return _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
}
YNN_ALWAYS_INLINE __m256d concat(__m128d lo, __m128d hi) {
  return _mm256_insertf128_pd(_mm256_castpd128_pd256(lo), hi, 1);
}
YNN_ALWAYS_INLINE __m256i concat(__m128i lo, __m128i hi) {
  return _mm256_castps_si256(
      concat(_mm_castsi128_ps(lo), _mm_castsi128_ps(hi)));
}

}  // namespace internal

template <>
struct vec<double, 4> {
  using value_type = double;
  static constexpr std::integral_constant<size_t, 4> N = {};

  vec() = default;
  explicit vec(__m256d v) : v(v) {}
  vec(f64x2 lo, f64x2 hi) : v(internal::concat(lo.v, hi.v)) {}
  vec(double x) : v(_mm256_set1_pd(x)) {}  // NOLINT

  __m256d v;
};

template <>
struct vec<float, 8> {
  using value_type = float;
  static constexpr std::integral_constant<size_t, 8> N = {};

  vec() = default;
  explicit vec(__m256 v) : v(v) {}
  vec(f32x4 lo, f32x4 hi) : v(internal::concat(lo.v, hi.v)) {}
  vec(float x) : v(_mm256_set1_ps(x)) {}  // NOLINT

  __m256 v;
};

template <>
struct vec<uint32_t, 8> {
  using value_type = uint32_t;
  static constexpr std::integral_constant<size_t, 8> N = {};

  vec() = default;
  explicit vec(__m256i v) : v(v) {}
  vec(u32x4 lo, u32x4 hi) : v(internal::concat(lo.v, hi.v)) {}
  vec(uint32_t x) : v(_mm256_set1_epi32(x)) {}  // NOLINT

  __m256i v;
};

template <>
struct vec<int32_t, 8> {
  using value_type = int32_t;
  static constexpr std::integral_constant<size_t, 8> N = {};

  vec() = default;
  explicit vec(__m256i v) : v(v) {}
  vec(s32x4 lo, s32x4 hi) : v(internal::concat(lo.v, hi.v)) {}
  vec(int32_t x) : v(_mm256_set1_epi32(x)) {}  // NOLINT

  __m256i v;
};

template <>
struct vec<bfloat16, 16> {
  using value_type = bfloat16;
  static constexpr std::integral_constant<size_t, 16> N = {};

  vec() = default;
  explicit vec(__m256i v) : v(v) {}
  vec(bf16x8 lo, bf16x8 hi) : v(internal::concat(lo.v, hi.v)) {}
  vec(bfloat16 x) : v(_mm256_set1_epi16(x.to_bits())) {}  // NOLINT

  __m256i v;
};

template <>
struct vec<half, 16> {
  using value_type = half;
  static constexpr std::integral_constant<size_t, 16> N = {};

  vec() = default;
  explicit vec(__m256i v) : v(v) {}
  vec(f16x8 lo, f16x8 hi) : v(internal::concat(lo.v, hi.v)) {}
  vec(half x) : v(_mm256_set1_epi16(x.to_bits())) {}  // NOLINT

  __m256i v;
};

template <>
struct vec<uint16_t, 16> {
  using value_type = uint16_t;
  static constexpr std::integral_constant<size_t, 16> N = {};

  vec() = default;
  explicit vec(__m256i v) : v(v) {}
  vec(u16x8 lo, u16x8 hi) : v(internal::concat(lo.v, hi.v)) {}
  vec(uint16_t x) : v(_mm256_set1_epi16(x)) {}  // NOLINT

  __m256i v;
};

template <>
struct vec<int16_t, 16> {
  using value_type = int16_t;
  static constexpr std::integral_constant<size_t, 16> N = {};

  vec() = default;
  explicit vec(__m256i v) : v(v) {}
  vec(s16x8 lo, s16x8 hi) : v(internal::concat(lo.v, hi.v)) {}
  vec(int16_t x) : v(_mm256_set1_epi16(x)) {}  // NOLINT

  __m256i v;
};

template <>
struct vec<uint8_t, 32> {
  using value_type = uint8_t;
  static constexpr std::integral_constant<size_t, 32> N = {};

  vec() = default;
  explicit vec(__m256i v) : v(v) {}
  vec(u8x16 lo, u8x16 hi) : v(internal::concat(lo.v, hi.v)) {}
  vec(uint8_t x) : v(_mm256_set1_epi8(x)) {}  // NOLINT

  __m256i v;
};

template <>
struct vec<int8_t, 32> {
  using value_type = int8_t;
  static constexpr std::integral_constant<size_t, 32> N = {};

  vec() = default;
  explicit vec(__m256i v) : v(v) {}
  vec(s8x16 lo, s8x16 hi) : v(internal::concat(lo.v, hi.v)) {}
  vec(int8_t x) : v(_mm256_set1_epi8(x)) {}  // NOLINT

  __m256i v;
};

template <>
struct vec<int64_t, 4> {
  using value_type = int64_t;
  static constexpr std::integral_constant<size_t, 4> N = {};

  vec() = default;
  explicit vec(__m256i v) : v(v) {}
  vec(int64_t x) : v(_mm256_set1_epi64x(x)) {}  // NOLINT

  __m256i v;
};

using f64x4 = vec<double, 4>;
using f32x8 = vec<float, 8>;
using u32x8 = vec<uint32_t, 8>;
using s32x8 = vec<int32_t, 8>;
using bf16x16 = vec<bfloat16, 16>;
using f16x16 = vec<half, 16>;
using u16x16 = vec<uint16_t, 16>;
using s16x16 = vec<int16_t, 16>;
using u8x32 = vec<uint8_t, 32>;
using s8x32 = vec<int8_t, 32>;
using s64x4 = vec<int64_t, 4>;
using s64x2 = vec<int64_t, 2>;

struct s2x128 {
  __m256i v;
};

struct s4x64 {
  __m256i v;
};

YNN_ALWAYS_INLINE f64x2 lo(f64x4 x) { return f64x2{internal::lo(x.v)}; }
YNN_ALWAYS_INLINE f64x2 hi(f64x4 x) { return f64x2{internal::hi(x.v)}; }
YNN_ALWAYS_INLINE f32x4 lo(f32x8 x) { return f32x4{internal::lo(x.v)}; }
YNN_ALWAYS_INLINE f32x4 hi(f32x8 x) { return f32x4{internal::hi(x.v)}; }
YNN_ALWAYS_INLINE u32x4 lo(u32x8 x) { return u32x4{internal::lo(x.v)}; }
YNN_ALWAYS_INLINE u32x4 hi(u32x8 x) { return u32x4{internal::hi(x.v)}; }
YNN_ALWAYS_INLINE s32x4 lo(s32x8 x) { return s32x4{internal::lo(x.v)}; }
YNN_ALWAYS_INLINE s32x4 hi(s32x8 x) { return s32x4{internal::hi(x.v)}; }
YNN_ALWAYS_INLINE bf16x8 lo(bf16x16 x) { return bf16x8{internal::lo(x.v)}; }
YNN_ALWAYS_INLINE bf16x8 hi(bf16x16 x) { return bf16x8{internal::hi(x.v)}; }
YNN_ALWAYS_INLINE f16x8 lo(f16x16 x) { return f16x8{internal::lo(x.v)}; }
YNN_ALWAYS_INLINE f16x8 hi(f16x16 x) { return f16x8{internal::hi(x.v)}; }
YNN_ALWAYS_INLINE u16x8 lo(u16x16 x) { return u16x8{internal::lo(x.v)}; }
YNN_ALWAYS_INLINE u16x8 hi(u16x16 x) { return u16x8{internal::hi(x.v)}; }
YNN_ALWAYS_INLINE s16x8 lo(s16x16 x) { return s16x8{internal::lo(x.v)}; }
YNN_ALWAYS_INLINE s16x8 hi(s16x16 x) { return s16x8{internal::hi(x.v)}; }
YNN_ALWAYS_INLINE u8x16 lo(u8x32 x) { return u8x16{internal::lo(x.v)}; }
YNN_ALWAYS_INLINE u8x16 hi(u8x32 x) { return u8x16{internal::hi(x.v)}; }
YNN_ALWAYS_INLINE s8x16 lo(s8x32 x) { return s8x16{internal::lo(x.v)}; }
YNN_ALWAYS_INLINE s8x16 hi(s8x32 x) { return s8x16{internal::hi(x.v)}; }
YNN_ALWAYS_INLINE s64x2 lo(s64x4 x) { return s64x2{internal::lo(x.v)}; }
YNN_ALWAYS_INLINE s64x2 hi(s64x4 x) { return s64x2{internal::hi(x.v)}; }

namespace internal {

// These overloads are x86-specific helpers for implementing templated
// interleave/transpose helpers that are not x86-specific.
YNN_ALWAYS_INLINE f32x8 unpacklo(f32x8 a, f32x8 b) {
  return f32x8{_mm256_unpacklo_ps(a.v, b.v)};
}
YNN_ALWAYS_INLINE f32x8 unpackhi(f32x8 a, f32x8 b) {
  return f32x8{_mm256_unpackhi_ps(a.v, b.v)};
}

}  // namespace internal

YNN_ALWAYS_INLINE f64x4 load_aligned(const double* ptr, decltype(f64x4::N),
                                     f64x4 = {}) {
  return f64x4{_mm256_load_pd(ptr)};
}
YNN_ALWAYS_INLINE f32x8 load_aligned(const float* ptr, decltype(f32x8::N),
                                     f32x8 = {}) {
  return f32x8{_mm256_load_ps(ptr)};
}
YNN_ALWAYS_INLINE s32x8 load_aligned(const int32_t* ptr, decltype(s32x8::N),
                                     s32x8 = {}) {
  return s32x8{_mm256_load_si256(reinterpret_cast<const __m256i*>(ptr))};
}
YNN_ALWAYS_INLINE bf16x16 load_aligned(const bfloat16* ptr,
                                       decltype(bf16x16::N), bf16x16 = {}) {
  return bf16x16{_mm256_load_si256(reinterpret_cast<const __m256i*>(ptr))};
}
YNN_ALWAYS_INLINE f16x16 load_aligned(const half* ptr, decltype(f16x16::N),
                                      f16x16 = {}) {
  return f16x16{_mm256_load_si256(reinterpret_cast<const __m256i*>(ptr))};
}
YNN_ALWAYS_INLINE s16x16 load_aligned(const int16_t* ptr, decltype(s16x16::N),
                                      s16x16 = {}) {
  return s16x16{_mm256_load_si256(reinterpret_cast<const __m256i*>(ptr))};
}
YNN_ALWAYS_INLINE u8x32 load_aligned(const uint8_t* ptr, decltype(u8x32::N),
                                     u8x32 = {}) {
  return u8x32{_mm256_load_si256(reinterpret_cast<const __m256i*>(ptr))};
}
YNN_ALWAYS_INLINE s8x32 load_aligned(const int8_t* ptr, decltype(s8x32::N),
                                     s8x32 = {}) {
  return s8x32{_mm256_load_si256(reinterpret_cast<const __m256i*>(ptr))};
}

YNN_ALWAYS_INLINE void store_aligned(double* ptr, f64x4 b,
                                     decltype(f64x4::N) = {}) {
  _mm256_store_pd(ptr, b.v);
}
YNN_ALWAYS_INLINE void store_aligned(float* ptr, f32x8 b,
                                     decltype(f32x8::N) = {}) {
  _mm256_store_ps(ptr, b.v);
}
YNN_ALWAYS_INLINE void store_aligned(uint32_t* ptr, u32x8 b,
                                     decltype(u32x8::N) = {}) {
  _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store_aligned(int32_t* ptr, s32x8 b,
                                     decltype(s32x8::N) = {}) {
  _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store_aligned(bfloat16* ptr, bf16x16 b,
                                     decltype(bf16x16::N) = {}) {
  _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store_aligned(half* ptr, f16x16 b,
                                     decltype(f16x16::N) = {}) {
  _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store_aligned(uint16_t* ptr, u16x16 b,
                                     decltype(u16x16::N) = {}) {
  _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store_aligned(int16_t* ptr, s16x16 b,
                                     decltype(s16x16::N) = {}) {
  _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store_aligned(uint8_t* ptr, u8x32 b,
                                     decltype(u8x32::N) = {}) {
  _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store_aligned(int8_t* ptr, s8x32 b,
                                     decltype(s8x32::N) = {}) {
  _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), b.v);
}

YNN_ALWAYS_INLINE f64x4 load(const double* ptr, decltype(f64x4::N),
                             f64x4 = {}) {
  return f64x4{_mm256_loadu_pd(ptr)};
}
YNN_ALWAYS_INLINE f32x8 load(const float* ptr, decltype(f32x8::N), f32x8 = {}) {
  return f32x8{_mm256_loadu_ps(ptr)};
}
YNN_ALWAYS_INLINE s32x8 load(const int32_t* ptr, decltype(s32x8::N),
                             s32x8 = {}) {
  return s32x8{_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr))};
}
YNN_ALWAYS_INLINE bf16x16 load(const bfloat16* ptr, decltype(bf16x16::N),
                               bf16x16 = {}) {
  return bf16x16{_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr))};
}
YNN_ALWAYS_INLINE f16x16 load(const half* ptr, decltype(f16x16::N),
                              f16x16 = {}) {
  return f16x16{_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr))};
}
YNN_ALWAYS_INLINE u16x16 load(const uint16_t* ptr, decltype(u16x16::N),
                              u16x16 = {}) {
  return u16x16{_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr))};
}
YNN_ALWAYS_INLINE s16x16 load(const int16_t* ptr, decltype(s16x16::N),
                              s16x16 = {}) {
  return s16x16{_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr))};
}
YNN_ALWAYS_INLINE u8x32 load(const uint8_t* ptr, decltype(u8x32::N),
                             u8x32 = {}) {
  return u8x32{_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr))};
}
YNN_ALWAYS_INLINE s8x32 load(const int8_t* ptr, decltype(s8x32::N),
                             s8x32 = {}) {
  return s8x32{_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr))};
}

YNN_ALWAYS_INLINE void store(double* ptr, f64x4 b, decltype(f64x4::N) = {}) {
  _mm256_storeu_pd(ptr, b.v);
}
YNN_ALWAYS_INLINE void store(float* ptr, f32x8 b, decltype(f32x8::N) = {}) {
  _mm256_storeu_ps(ptr, b.v);
}
YNN_ALWAYS_INLINE void store(uint32_t* ptr, u32x8 b, decltype(u32x8::N) = {}) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x8 b, decltype(s32x8::N) = {}) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(bfloat16* ptr, bf16x16 b,
                             decltype(bf16x16::N) = {}) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(half* ptr, f16x16 b, decltype(f16x16::N) = {}) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(uint16_t* ptr, u16x16 b,
                             decltype(u16x16::N) = {}) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(int16_t* ptr, s16x16 b, decltype(s16x16::N) = {}) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x32 b, decltype(u8x32::N) = {}) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(int8_t* ptr, s8x32 b, decltype(s8x32::N) = {}) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), b.v);
}

#ifdef YNN_ARCH_X86_AVX512

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

#else  // YNN_ARCH_X86_AVX512

namespace internal {

// Align this to avoid spanning a cache line.
alignas(64) static constexpr int32_t mask_table[16] = {
    -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};

YNN_ALWAYS_INLINE f64x4 maskload(const double* ptr, __m256i mask) {
  return f64x4{_mm256_maskload_pd(ptr, mask)};
}
YNN_ALWAYS_INLINE f32x8 maskload(const float* ptr, __m256i mask) {
  return f32x8{_mm256_maskload_ps(ptr, mask)};
}
YNN_ALWAYS_INLINE s32x8 maskload(const int32_t* ptr, __m256i mask) {
  return s32x8{_mm256_castps_si256(
      _mm256_maskload_ps(reinterpret_cast<const float*>(ptr), mask))};
}

YNN_ALWAYS_INLINE f64x4 maskload(const double* ptr, __m256d src, __m256i mask) {
  return f64x4{_mm256_blendv_pd(src, _mm256_maskload_pd(ptr, mask),
                                _mm256_castsi256_pd(mask))};
}
YNN_ALWAYS_INLINE f32x8 maskload(const float* ptr, __m256 src, __m256i mask) {
  return f32x8{_mm256_blendv_ps(src, _mm256_maskload_ps(ptr, mask),
                                _mm256_castsi256_ps(mask))};
}
YNN_ALWAYS_INLINE s32x8 maskload(const int32_t* ptr, __m256i src,
                                 __m256i mask) {
  return s32x8{_mm256_castps_si256(_mm256_blendv_ps(
      _mm256_castsi256_ps(src),
      _mm256_maskload_ps(reinterpret_cast<const float*>(ptr), mask),
      _mm256_castsi256_ps(mask)))};
}

YNN_ALWAYS_INLINE void maskstore(double* ptr, f64x4 val, __m256i mask) {
  _mm256_maskstore_pd(ptr, mask, val.v);
}
YNN_ALWAYS_INLINE void maskstore(float* ptr, f32x8 val, __m256i mask) {
  _mm256_maskstore_ps(ptr, mask, val.v);
}
YNN_ALWAYS_INLINE void maskstore(int32_t* ptr, s32x8 val, __m256i mask) {
  _mm256_maskstore_ps(reinterpret_cast<float*>(ptr), mask,
                      _mm256_castsi256_ps(val.v));
}

// Partial load/store with a non-constant number of elements.
template <typename T>
YNN_ALWAYS_INLINE vec<T, 4> partial_load_mask_x64x4(const T* ptr, size_t n,
                                                    vec<T, 4> src) {
  assert(n <= 4);
  auto mask = _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>(&mask_table[8 - n * 2]));
  return vec<T, 4>{maskload(ptr, src.v, mask)};
}

// Partial load/store with a non-constant number of elements.
template <typename T>
YNN_ALWAYS_INLINE vec<T, 4> partial_load_mask_x64x4(const T* ptr, size_t n,
                                                    zeros<4>) {
  assert(n <= 4);
  auto mask = _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>(&mask_table[8 - n * 2]));
  return vec<T, 4>{maskload(ptr, mask)};
}

template <typename T>
YNN_ALWAYS_INLINE vec<T, 8> partial_load_mask_x32x8(const T* ptr, size_t n,
                                                    vec<T, 8> src) {
  assert(n <= 8);
  auto mask =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&mask_table[8 - n]));
  return vec<T, 8>{maskload(ptr, src.v, mask)};
}

// Partial load/store with a non-constant number of elements.
template <typename T>
YNN_ALWAYS_INLINE vec<T, 8> partial_load_mask_x32x8(const T* ptr, size_t n,
                                                    zeros<8>) {
  assert(n <= 8);
  auto mask =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&mask_table[8 - n]));
  return vec<T, 8>{maskload(ptr, mask)};
}

template <typename T>
YNN_ALWAYS_INLINE void partial_store_x64x4(T* ptr, vec<T, 4> val, size_t n) {
  assert(n <= 4);
  auto mask = _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>(&mask_table[8 - n * 2]));
  maskstore(ptr, val, mask);
}

template <typename T>
YNN_ALWAYS_INLINE void partial_store_x32x8(T* ptr, vec<T, 8> val, size_t n) {
  assert(n <= 8);
  auto mask =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&mask_table[8 - n]));
  maskstore(ptr, val, mask);
}

}  // namespace internal

YNN_ALWAYS_INLINE f64x4 load(const double* ptr, size_t n, f64x4 src) {
  return internal::partial_load_mask_x64x4(ptr, n, src);
}
YNN_ALWAYS_INLINE f32x8 load(const float* ptr, size_t n, f32x8 src) {
  return internal::partial_load_mask_x32x8(ptr, n, src);
}
YNN_ALWAYS_INLINE s32x8 load(const int32_t* ptr, size_t n, s32x8 src) {
  return internal::partial_load_mask_x32x8(ptr, n, src);
}

YNN_ALWAYS_INLINE f64x4 load(const double* ptr, size_t n, zeros<4> src) {
  return internal::partial_load_mask_x64x4(ptr, n, src);
}
YNN_ALWAYS_INLINE f32x8 load(const float* ptr, size_t n, zeros<8> src) {
  return internal::partial_load_mask_x32x8(ptr, n, src);
}
YNN_ALWAYS_INLINE s32x8 load(const int32_t* ptr, size_t n, zeros<8> src) {
  return internal::partial_load_mask_x32x8(ptr, n, src);
}

YNN_ALWAYS_INLINE f64x4 load(const double* ptr, size_t n, undef<4> src) {
  return internal::partial_load_mask_x64x4(ptr, n, zeros<4>{});
}
YNN_ALWAYS_INLINE f32x8 load(const float* ptr, size_t n, undef<8> src) {
  return internal::partial_load_mask_x32x8(ptr, n, zeros<8>{});
}
YNN_ALWAYS_INLINE s32x8 load(const int32_t* ptr, size_t n, undef<8> src) {
  return internal::partial_load_mask_x32x8(ptr, n, zeros<8>{});
}

YNN_ALWAYS_INLINE void store(double* ptr, f64x4 val, size_t n) {
  internal::partial_store_x64x4(ptr, val, n);
}
YNN_ALWAYS_INLINE void store(float* ptr, f32x8 val, size_t n) {
  internal::partial_store_x32x8(ptr, val, n);
}
YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x8 val, size_t n) {
  internal::partial_store_x32x8(ptr, val, n);
}

#endif  // YNN_ARCH_X86_AVX512

YNN_ALWAYS_INLINE f64x4 operator+(f64x4 a, f64x4 b) {
  return f64x4{_mm256_add_pd(a.v, b.v)};
}
YNN_ALWAYS_INLINE f32x8 operator+(f32x8 a, f32x8 b) {
  return f32x8{_mm256_add_ps(a.v, b.v)};
}
YNN_ALWAYS_INLINE f64x4 operator-(f64x4 a, f64x4 b) {
  return f64x4{_mm256_sub_pd(a.v, b.v)};
}
YNN_ALWAYS_INLINE f32x8 operator-(f32x8 a, f32x8 b) {
  return f32x8{_mm256_sub_ps(a.v, b.v)};
}
YNN_ALWAYS_INLINE f64x4 operator*(f64x4 a, f64x4 b) {
  return f64x4{_mm256_mul_pd(a.v, b.v)};
}
YNN_ALWAYS_INLINE f32x8 operator*(f32x8 a, f32x8 b) {
  return f32x8{_mm256_mul_ps(a.v, b.v)};
}
YNN_ALWAYS_INLINE f64x4 operator/(f64x4 a, f64x4 b) {
  return f64x4{_mm256_div_pd(a.v, b.v)};
}
YNN_ALWAYS_INLINE f32x8 operator/(f32x8 a, f32x8 b) {
  return f32x8{_mm256_div_ps(a.v, b.v)};
}

#ifdef YNN_ARCH_X86_AVX2
YNN_ALWAYS_INLINE s32x8 operator+(s32x8 a, s32x8 b) {
  return s32x8{_mm256_add_epi32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x32 operator+(s8x32 a, s8x32 b) {
  return s8x32{_mm256_add_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x32 operator+(u8x32 a, u8x32 b) {
  return u8x32{_mm256_add_epi8(a.v, b.v)};
}

YNN_ALWAYS_INLINE s32x8 operator-(s32x8 a, s32x8 b) {
  return s32x8{_mm256_sub_epi32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x32 operator-(s8x32 a, s8x32 b) {
  return s8x32{_mm256_sub_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x32 operator-(u8x32 a, u8x32 b) {
  return u8x32{_mm256_sub_epi8(a.v, b.v)};
}

YNN_ALWAYS_INLINE s32x8 operator*(s32x8 a, s32x8 b) {
  return s32x8{_mm256_mullo_epi32(a.v, b.v)};
}

YNN_ALWAYS_INLINE s16x16 add_sat(s16x16 a, s16x16 b) {
  return s16x16{_mm256_adds_epi16(a.v, b.v)};
}
YNN_ALWAYS_INLINE u16x16 add_sat(u16x16 a, u16x16 b) {
  return u16x16{_mm256_adds_epu16(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x32 add_sat(s8x32 a, s8x32 b) {
  return s8x32{_mm256_adds_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x32 add_sat(u8x32 a, u8x32 b) {
  return u8x32{_mm256_adds_epu8(a.v, b.v)};
}

YNN_ALWAYS_INLINE s16x16 sub_sat(s16x16 a, s16x16 b) {
  return s16x16{_mm256_subs_epi16(a.v, b.v)};
}
YNN_ALWAYS_INLINE u16x16 sub_sat(u16x16 a, u16x16 b) {
  return u16x16{_mm256_subs_epu16(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x32 sub_sat(s8x32 a, s8x32 b) {
  return s8x32{_mm256_subs_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x32 sub_sat(u8x32 a, u8x32 b) {
  return u8x32{_mm256_subs_epu8(a.v, b.v)};
}
#endif  // YNN_ARCH_X86_AVX2

YNN_ALWAYS_INLINE f64x4 min(f64x4 a, f64x4 b) {
  return f64x4{_mm256_min_pd(a.v, b.v)};
}
YNN_ALWAYS_INLINE f32x8 min(f32x8 a, f32x8 b) {
  return f32x8{_mm256_min_ps(a.v, b.v)};
}
YNN_ALWAYS_INLINE f64x4 max(f64x4 a, f64x4 b) {
  return f64x4{_mm256_max_pd(a.v, b.v)};
}
YNN_ALWAYS_INLINE f32x8 max(f32x8 a, f32x8 b) {
  return f32x8{_mm256_max_ps(a.v, b.v)};
}

#ifdef YNN_ARCH_X86_AVX2
YNN_ALWAYS_INLINE s32x8 min(s32x8 a, s32x8 b) {
  return s32x8{_mm256_min_epi32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x16 min(s16x16 a, s16x16 b) {
  return s16x16{_mm256_min_epi16(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x32 min(s8x32 a, s8x32 b) {
  return s8x32{_mm256_min_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x32 min(u8x32 a, u8x32 b) {
  return u8x32{_mm256_min_epu8(a.v, b.v)};
}

YNN_ALWAYS_INLINE s32x8 max(s32x8 a, s32x8 b) {
  return s32x8{_mm256_max_epi32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x16 max(s16x16 a, s16x16 b) {
  return s16x16{_mm256_max_epi16(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x32 max(s8x32 a, s8x32 b) {
  return s8x32{_mm256_max_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x32 max(u8x32 a, u8x32 b) {
  return u8x32{_mm256_max_epu8(a.v, b.v)};
}
#endif  // YNN_ARCH_X86_AVX2

YNN_ALWAYS_INLINE f64x4 floor(f64x4 a) { return f64x4{_mm256_floor_pd(a.v)}; }
YNN_ALWAYS_INLINE f64x4 ceil(f64x4 a) { return f64x4{_mm256_ceil_pd(a.v)}; }
YNN_ALWAYS_INLINE f64x4 round(f64x4 a) {
  return f64x4{
      _mm256_round_pd(a.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)};
}
YNN_ALWAYS_INLINE f64x4 sqrt(f64x4 a) { return f64x4{_mm256_sqrt_pd(a.v)}; }
YNN_ALWAYS_INLINE f64x4 abs(f64x4 a) {
  return f64x4{_mm256_and_pd(
      a.v, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF)))};
}
YNN_ALWAYS_INLINE f32x8 floor(f32x8 a) { return f32x8{_mm256_floor_ps(a.v)}; }
YNN_ALWAYS_INLINE f32x8 ceil(f32x8 a) { return f32x8{_mm256_ceil_ps(a.v)}; }
YNN_ALWAYS_INLINE f32x8 round(f32x8 a) {
  return f32x8{
      _mm256_round_ps(a.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)};
}
YNN_ALWAYS_INLINE f32x8 sqrt(f32x8 a) { return f32x8{_mm256_sqrt_ps(a.v)}; }
YNN_ALWAYS_INLINE f32x8 abs(f32x8 a) {
  return f32x8{_mm256_and_ps(a.v, _mm256_set1_ps(bit_cast<float>(0x7FFFFFFF)))};
}
#ifdef YNN_ARCH_X86_AVX2
YNN_ALWAYS_INLINE u8x32 abs(s8x32 a) { return u8x32{_mm256_abs_epi8(a.v)}; }
YNN_ALWAYS_INLINE u16x16 abs(s16x16 a) { return u16x16{_mm256_abs_epi16(a.v)}; }
YNN_ALWAYS_INLINE u32x8 abs(s32x8 a) { return u32x8{_mm256_abs_epi32(a.v)}; }
#endif  // YNN_ARCH_X86_AVX2

YNN_ALWAYS_INLINE s32x8 operator&(s32x8 a, s32x8 b) {
  return s32x8{_mm256_castps_si256(
      _mm256_and_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v)))};
}
YNN_ALWAYS_INLINE s32x8 operator|(s32x8 a, s32x8 b) {
  return s32x8{_mm256_castps_si256(
      _mm256_or_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v)))};
}
YNN_ALWAYS_INLINE s32x8 operator^(s32x8 a, s32x8 b) {
  return s32x8{_mm256_castps_si256(
      _mm256_xor_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v)))};
}
YNN_ALWAYS_INLINE s32x8 operator~(s32x8 a) {
  return s32x8{_mm256_castps_si256(_mm256_xor_ps(
      _mm256_castsi256_ps(a.v), _mm256_set1_ps(bit_cast<float>(-1))))};
}

YNN_ALWAYS_INLINE s64x4 operator&(s64x4 a, s64x4 b) {
  return s64x4{_mm256_castps_si256(
      _mm256_and_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v)))};
}
YNN_ALWAYS_INLINE s64x4 operator|(s64x4 a, s64x4 b) {
  return s64x4{_mm256_castps_si256(
      _mm256_or_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v)))};
}
YNN_ALWAYS_INLINE s64x4 operator^(s64x4 a, s64x4 b) {
  return s64x4{_mm256_castps_si256(
      _mm256_xor_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v)))};
}
YNN_ALWAYS_INLINE s64x4 operator~(s64x4 a) {
  return s64x4{_mm256_castps_si256(_mm256_xor_ps(
      _mm256_castsi256_ps(a.v), _mm256_set1_ps(bit_cast<float>(-1))))};
}

YNN_ALWAYS_INLINE s16x16 operator&(s16x16 a, s16x16 b) {
  return s16x16{_mm256_castps_si256(
      _mm256_and_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v)))};
}
YNN_ALWAYS_INLINE s16x16 operator|(s16x16 a, s16x16 b) {
  return s16x16{_mm256_castps_si256(
      _mm256_or_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v)))};
}
YNN_ALWAYS_INLINE s16x16 operator^(s16x16 a, s16x16 b) {
  return s16x16{_mm256_castps_si256(
      _mm256_xor_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v)))};
}
YNN_ALWAYS_INLINE s16x16 operator~(s16x16 a) {
  return s16x16{_mm256_castps_si256(_mm256_xor_ps(
      _mm256_castsi256_ps(a.v), _mm256_set1_ps(bit_cast<float>(-1))))};
}

YNN_ALWAYS_INLINE u8x32 operator&(u8x32 a, u8x32 b) {
  return u8x32{_mm256_castps_si256(
      _mm256_and_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v)))};
}
YNN_ALWAYS_INLINE u8x32 operator|(u8x32 a, u8x32 b) {
  return u8x32{_mm256_castps_si256(
      _mm256_or_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v)))};
}
YNN_ALWAYS_INLINE u8x32 operator^(u8x32 a, u8x32 b) {
  return u8x32{_mm256_castps_si256(
      _mm256_xor_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v)))};
}
YNN_ALWAYS_INLINE u8x32 operator~(u8x32 a) {
  return u8x32{_mm256_castps_si256(_mm256_xor_ps(
      _mm256_castsi256_ps(a.v), _mm256_set1_ps(bit_cast<float>(-1))))};
}

YNN_ALWAYS_INLINE s8x32 operator&(s8x32 a, s8x32 b) {
  return s8x32{_mm256_castps_si256(
      _mm256_and_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v)))};
}
YNN_ALWAYS_INLINE s8x32 operator|(s8x32 a, s8x32 b) {
  return s8x32{_mm256_castps_si256(
      _mm256_or_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v)))};
}
YNN_ALWAYS_INLINE s8x32 operator^(s8x32 a, s8x32 b) {
  return s8x32{_mm256_castps_si256(
      _mm256_xor_ps(_mm256_castsi256_ps(a.v), _mm256_castsi256_ps(b.v)))};
}
YNN_ALWAYS_INLINE s8x32 operator~(s8x32 a) {
  return s8x32{_mm256_castps_si256(_mm256_xor_ps(
      _mm256_castsi256_ps(a.v), _mm256_set1_ps(bit_cast<float>(-1))))};
}

#ifdef YNN_ARCH_X86_AVX2
YNN_ALWAYS_INLINE s16x16 operator>>(s16x16 a, int b) {
  return s16x16{_mm256_srai_epi16(a.v, b)};
}
YNN_ALWAYS_INLINE s16x16 operator<<(s16x16 a, int b) {
  return s16x16{_mm256_slli_epi16(a.v, b)};
}
YNN_ALWAYS_INLINE s32x8 operator<<(s32x8 a, int b) {
  return s32x8{_mm256_slli_epi32(a.v, b)};
}
#endif  // YNN_ARCH_X86_AVX2

YNN_ALWAYS_INLINE s32x8 operator==(f32x8 a, f32x8 b) {
  return s32x8{_mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_EQ_OQ))};
}
YNN_ALWAYS_INLINE s32x8 operator!=(f32x8 a, f32x8 b) {
  return s32x8{_mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_NEQ_OQ))};
}
YNN_ALWAYS_INLINE s32x8 operator<(f32x8 a, f32x8 b) {
  return s32x8{_mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_LT_OQ))};
}
YNN_ALWAYS_INLINE s32x8 operator<=(f32x8 a, f32x8 b) {
  return s32x8{_mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_LE_OQ))};
}
YNN_ALWAYS_INLINE s32x8 operator>(f32x8 a, f32x8 b) {
  return s32x8{_mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_GT_OQ))};
}
YNN_ALWAYS_INLINE s32x8 operator>=(f32x8 a, f32x8 b) {
  return s32x8{_mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_GE_OQ))};
}

YNN_ALWAYS_INLINE s64x4 operator==(f64x4 a, f64x4 b) {
  return s64x4{_mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_EQ_OQ))};
}
YNN_ALWAYS_INLINE s64x4 operator!=(f64x4 a, f64x4 b) {
  return s64x4{_mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_NEQ_OQ))};
}
YNN_ALWAYS_INLINE s64x4 operator<(f64x4 a, f64x4 b) {
  return s64x4{_mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_LT_OQ))};
}
YNN_ALWAYS_INLINE s64x4 operator<=(f64x4 a, f64x4 b) {
  return s64x4{_mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_LE_OQ))};
}
YNN_ALWAYS_INLINE s64x4 operator>(f64x4 a, f64x4 b) {
  return s64x4{_mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_GT_OQ))};
}
YNN_ALWAYS_INLINE s64x4 operator>=(f64x4 a, f64x4 b) {
  return s64x4{_mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_GE_OQ))};
}

#ifdef YNN_ARCH_X86_AVX2
YNN_ALWAYS_INLINE s32x8 operator==(s32x8 a, s32x8 b) {
  return s32x8{_mm256_cmpeq_epi32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x8 operator>(s32x8 a, s32x8 b) {
  return s32x8{_mm256_cmpgt_epi32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x8 operator<(s32x8 a, s32x8 b) { return b > a; }

YNN_ALWAYS_INLINE s16x16 operator==(s16x16 a, s16x16 b) {
  return s16x16{_mm256_cmpeq_epi16(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x16 operator>(s16x16 a, s16x16 b) {
  return s16x16{_mm256_cmpgt_epi16(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x16 operator<(s16x16 a, s16x16 b) { return b > a; }

YNN_ALWAYS_INLINE s8x32 operator==(s8x32 a, s8x32 b) {
  return s8x32{_mm256_cmpeq_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x32 operator>(s8x32 a, s8x32 b) {
  return s8x32{_mm256_cmpgt_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x32 operator<(s8x32 a, s8x32 b) { return b > a; }
#endif  // YNN_ARCH_X86_AVX2

YNN_ALWAYS_INLINE s32x8 isnan(f32x8 a) {
  return s32x8{_mm256_castps_si256(_mm256_cmp_ps(a.v, a.v, _CMP_UNORD_Q))};
}
YNN_ALWAYS_INLINE s64x4 isnan(f64x4 a) {
  return s64x4{_mm256_castpd_si256(_mm256_cmp_pd(a.v, a.v, _CMP_UNORD_Q))};
}

YNN_ALWAYS_INLINE s32x8 isinf(f32x8 a) {
  __m256 mask = _mm256_set1_ps(bit_cast<float>(0x7FFFFFFF));
  __m256 inf = _mm256_set1_ps(bit_cast<float>(0x7F800000));
  return s32x8{_mm256_castps_si256(
      _mm256_cmp_ps(_mm256_and_ps(a.v, mask), inf, _CMP_EQ_OQ))};
}
YNN_ALWAYS_INLINE s64x4 isinf(f64x4 a) {
  __m256d mask = _mm256_set1_pd(bit_cast<double>(0x7FFFFFFFFFFFFFFFULL));
  __m256d inf = _mm256_set1_pd(bit_cast<double>(0x7FF0000000000000ULL));
  return s64x4{_mm256_castpd_si256(
      _mm256_cmp_pd(_mm256_and_pd(a.v, mask), inf, _CMP_EQ_OQ))};
}

YNN_ALWAYS_INLINE s32x8 isfinite(f32x8 a) {
  __m256 mask = _mm256_set1_ps(bit_cast<float>(0x7FFFFFFF));
  __m256 inf = _mm256_set1_ps(bit_cast<float>(0x7F800000));
  return s32x8{_mm256_castps_si256(
      _mm256_cmp_ps(_mm256_and_ps(a.v, mask), inf, _CMP_LT_OQ))};
}
YNN_ALWAYS_INLINE s64x4 isfinite(f64x4 a) {
  __m256d mask = _mm256_set1_pd(bit_cast<double>(0x7FFFFFFFFFFFFFFFULL));
  __m256d inf = _mm256_set1_pd(bit_cast<double>(0x7FF0000000000000ULL));
  return s64x4{_mm256_castpd_si256(
      _mm256_cmp_pd(_mm256_and_pd(a.v, mask), inf, _CMP_LT_OQ))};
}

YNN_ALWAYS_INLINE f32x8 select(s32x8 cond, f32x8 a, f32x8 b) {
  return f32x8{_mm256_blendv_ps(b.v, a.v, _mm256_castsi256_ps(cond.v))};
}
YNN_ALWAYS_INLINE f64x4 select(s64x4 cond, f64x4 a, f64x4 b) {
  return f64x4{_mm256_blendv_pd(b.v, a.v, _mm256_castsi256_pd(cond.v))};
}

#ifdef YNN_ARCH_X86_AVX2
YNN_ALWAYS_INLINE s32x8 select(s32x8 cond, s32x8 a, s32x8 b) {
  return s32x8{_mm256_blendv_epi8(b.v, a.v, cond.v)};
}
YNN_ALWAYS_INLINE u32x8 select(s32x8 cond, u32x8 a, u32x8 b) {
  return u32x8{_mm256_blendv_epi8(b.v, a.v, cond.v)};
}
YNN_ALWAYS_INLINE s16x16 select(s16x16 cond, s16x16 a, s16x16 b) {
  return s16x16{_mm256_blendv_epi8(b.v, a.v, cond.v)};
}
YNN_ALWAYS_INLINE u16x16 select(s16x16 cond, u16x16 a, u16x16 b) {
  return u16x16{_mm256_blendv_epi8(b.v, a.v, cond.v)};
}
YNN_ALWAYS_INLINE s8x32 select(s8x32 cond, s8x32 a, s8x32 b) {
  return s8x32{_mm256_blendv_epi8(b.v, a.v, cond.v)};
}
YNN_ALWAYS_INLINE u8x32 select(s8x32 cond, u8x32 a, u8x32 b) {
  return u8x32{_mm256_blendv_epi8(b.v, a.v, cond.v)};
}
#else   // YNN_ARCH_X86_AVX2
YNN_ALWAYS_INLINE s32x8 select(s32x8 cond, s32x8 a, s32x8 b) {
  __m256 mc = _mm256_castsi256_ps(cond.v);
  __m256 ma = _mm256_castsi256_ps(a.v);
  __m256 mb = _mm256_castsi256_ps(b.v);
  return s32x8{_mm256_castps_si256(_mm256_blendv_ps(mb, ma, mc))};
}
#endif  // YNN_ARCH_X86_AVX2

YNN_ALWAYS_INLINE f64x4 fma(f64x4 a, f64x4 b, f64x4 acc) {
#ifdef YNN_ARCH_X86_FMA3
  return f64x4{_mm256_fmadd_pd(a.v, b.v, acc.v)};
#else
  return a * b + acc;
#endif
}
YNN_ALWAYS_INLINE f32x8 fma(f32x8 a, f32x8 b, f32x8 acc) {
#ifdef YNN_ARCH_X86_FMA3
  return f32x8{_mm256_fmadd_ps(a.v, b.v, acc.v)};
#else
  return a * b + acc;
#endif
}

YNN_ALWAYS_INLINE f64x4 cast(f32x4 a, double) {
  return f64x4{_mm256_cvtps_pd(a.v)};
}
YNN_ALWAYS_INLINE f32x4 cast(f64x4 a, float) {
  return f32x4{_mm256_cvtpd_ps(a.v)};
}

#ifdef YNN_ARCH_X86_F16C
YNN_ALWAYS_INLINE f32x8 cast(f16x8 a, float) {
  return f32x8{_mm256_cvtph_ps(a.v)};
}
YNN_ALWAYS_INLINE f16x8 cast(f32x8 a, half) {
  return f16x8{_mm256_cvtps_ph(a.v, _MM_FROUND_TO_NEAREST_INT)};
}
#endif  // YNN_ARCH_X86_F16C

#ifdef YNN_ARCH_X86_AVX2
YNN_ALWAYS_INLINE f32x8 exp2_round(f32x8 a) {
  const __m256 magic = _mm256_set1_ps(127.0f + static_cast<float>(1 << 23));
  const __m256 res_bits = _mm256_add_ps(a.v, magic);
  return f32x8{_mm256_castsi256_ps(
      _mm256_slli_epi32(_mm256_castps_si256(res_bits), 23))};
}
YNN_ALWAYS_INLINE f64x4 exp2_round(f64x4 a) {
  const __m256d magic = _mm256_set1_pd(1023.0 + static_cast<double>(1ll << 52));
  const __m256d res_bits = _mm256_add_pd(a.v, magic);
  return f64x4{_mm256_castsi256_pd(
      _mm256_slli_epi64(_mm256_castpd_si256(res_bits), 52))};
}
#endif  // YNN_ARCH_X86_AVX2

#ifdef YNN_ARCH_X86_AVX512
YNN_ALWAYS_INLINE f32x8 floor_log2(f32x8 a) {
  // getexp handles 0 correctly, but not negative numbers.
  __m256 res = _mm256_getexp_ps(a.v);
  __mmask8 negative = _mm256_cmp_ps_mask(a.v, _mm256_setzero_ps(), _CMP_LT_OQ);
  return f32x8{_mm256_mask_blend_ps(
      negative, res, _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN()))};
}
YNN_ALWAYS_INLINE f64x4 floor_log2(f64x4 a) {
  // getexp handles 0 correctly, but not negative numbers.
  __m256d res = _mm256_getexp_pd(a.v);
  __mmask8 negative = _mm256_cmp_pd_mask(a.v, _mm256_setzero_pd(), _CMP_LT_OQ);
  return f64x4{_mm256_mask_blend_pd(
      negative, res, _mm256_set1_pd(std::numeric_limits<double>::quiet_NaN()))};
}
#elif defined(YNN_ARCH_X86_AVX2)
YNN_ALWAYS_INLINE f32x8 floor_log2(f32x8 a) {
  __m256 sign_mask = _mm256_set1_ps(-0.0f);
  __m256 is_zero = _mm256_cmp_ps(a.v, _mm256_setzero_ps(), _CMP_EQ_OQ);
  a.v = _mm256_or_ps(_mm256_and_ps(is_zero, sign_mask), a.v);

  __m256i sign_and_exp_mask = _mm256_set1_epi32(0xFF800000);
  __m256i exp = _mm256_and_si256(_mm256_castps_si256(a.v), sign_and_exp_mask);

  __m256 infinity = _mm256_set1_ps(std::numeric_limits<float>::infinity());
  __m256 is_inf = _mm256_cmp_ps(a.v, infinity, _CMP_EQ_OQ);

  exp = _mm256_srai_epi32(exp, 8);

  __m256 bias_256 = _mm256_set1_ps(256.0f);
  __m256 bias_383 = _mm256_set1_ps(383.0f);
  __m256 res =
      _mm256_sub_ps(_mm256_or_ps(bias_256, _mm256_castsi256_ps(exp)), bias_383);
  return f32x8{_mm256_blendv_ps(res, infinity, is_inf)};
}
YNN_ALWAYS_INLINE f64x4 floor_log2(f64x4 a) {
  __m256d sign_mask = _mm256_set1_pd(-0.0);
  __m256d is_zero = _mm256_cmp_pd(a.v, _mm256_setzero_pd(), _CMP_EQ_OQ);
  a.v = _mm256_or_pd(_mm256_and_pd(is_zero, sign_mask), a.v);

  __m256i sign_and_exp_mask = _mm256_set1_epi64x(0xFFF0000000000000);
  __m256i exp = _mm256_and_si256(_mm256_castpd_si256(a.v), sign_and_exp_mask);

  __m256d infinity = _mm256_set1_pd(std::numeric_limits<double>::infinity());
  __m256d is_inf = _mm256_cmp_pd(a.v, infinity, _CMP_EQ_OQ);

  exp = _mm256_srai_epi32(exp, 11);

  __m256d bias_2048 = _mm256_set1_pd(2048.0);
  __m256d bias_3071 = _mm256_set1_pd(3071.0);
  __m256d res = _mm256_sub_pd(_mm256_or_pd(bias_2048, _mm256_castsi256_pd(exp)),
                              bias_3071);
  return f64x4{_mm256_blendv_pd(res, infinity, is_inf)};
}
#endif  // YNN_ARCH_X86_AVX2

#ifdef YNN_ARCH_X86_AVX2
YNN_ALWAYS_INLINE s32x8 cast(f32x8 f, int32_t) {
  const __m256 threshold = _mm256_set1_ps(2147483520.0f);
  const __m256 mask = _mm256_cmp_ps(f.v, threshold, _CMP_GT_OQ);
  const __m256 rounded =
      _mm256_round_ps(f.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  const __m256i res = _mm256_cvttps_epi32(rounded);
  return s32x8{_mm256_blendv_epi8(res, _mm256_set1_epi32(0x7fffffff),
                                  _mm256_castps_si256(mask))};
}

YNN_ALWAYS_INLINE f32x8 cast(s32x8 x, float) {
  return f32x8{_mm256_cvtepi32_ps(x.v)};
}

YNN_ALWAYS_INLINE f32x8 cast(bf16x8 a, float) {
  return f32x8{
      _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(a.v), 16))};
}

YNN_ALWAYS_INLINE s8x32 cast(s2x32 from, int8_t) {
  // 1. Broadcast the 64-bit GPR directly across the entire 256-bit register.
  __m256i dup = _mm256_set1_epi64x(static_cast<long long>(from.v));

  // 2. Duplicate the bytes so each 2-bit value has its own 8-bit lane.
  const __m256i mask_dup =
      _mm256_set_epi8(7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3,
                      3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  dup = _mm256_shuffle_epi8(dup, mask_dup);

  // 3. The cross-byte spill trick.
  __m256i shifted = _mm256_srli_epi32(dup, 4);
  __m256i blended = _mm256_blend_epi16(dup, shifted, 0xAA);
  __m256i masked = _mm256_and_si256(blended, _mm256_set1_epi32(0x0C030C03));

  // 4. Final sign-extension LUT
  const __m256i lut =
      _mm256_set_epi8(0, 0, 0, -1, 0, 0, 0, -2, 0, 0, 0, 1, -1, -2, 1, 0, 0, 0,
                      0, -1, 0, 0, 0, -2, 0, 0, 0, 1, -1, -2, 1, 0);

  return s8x32{_mm256_shuffle_epi8(lut, masked)};
}

YNN_ALWAYS_INLINE s8x32 cast(s4x32 from, int8_t) {
  // 1. Broadcast the 128-bit input to both lanes of the 256-bit register.
  __m256i dup = _mm256_broadcastsi128_si256(from.v);

  // 2. Duplicate each byte 2 times inside each 128-bit lane.
  const __m256i mask_dup =
      _mm256_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9,
                       9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15);
  dup = _mm256_shuffle_epi8(dup, mask_dup);

  // 3. Shift right and mask-blend bytes
  __m256i shifted = _mm256_srli_epi16(dup, 4);

  __m256i sel0 = _mm256_setr_epi8(
      0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0,
      0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0);
  __m256i sel1 = _mm256_setr_epi8(
      0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0,
      0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff);
  __m256i blended = _mm256_or_si256(_mm256_and_si256(dup, sel0),
                                    _mm256_and_si256(shifted, sel1));

  __m256i indices = _mm256_and_si256(blended, _mm256_set1_epi8(0x0f));
  const __m256i lut = _mm256_broadcastsi128_si256(
      _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1));

  return s8x32{_mm256_shuffle_epi8(lut, indices)};
}
#endif  // YNN_ARCH_X86_AVX2

#ifdef YNN_ARCH_X86_AVX2
YNN_ALWAYS_INLINE std::tuple<u8x32, u8x32> interleave(
    std::integral_constant<size_t, 128>, u8x32 x0, u8x32 x1) {
  return {u8x32{_mm256_permute2x128_si256(x0.v, x1.v, 32)},
          u8x32{_mm256_permute2x128_si256(x0.v, x1.v, 49)}};
}
YNN_ALWAYS_INLINE std::tuple<u8x32, u8x32> interleave(
    std::integral_constant<size_t, 64>, u8x32 x0, u8x32 x1) {
  return interleave(std::integral_constant<size_t, 128>{},
                    u8x32{_mm256_unpacklo_epi64(x0.v, x1.v)},
                    u8x32{_mm256_unpackhi_epi64(x0.v, x1.v)});
}
YNN_ALWAYS_INLINE std::tuple<u8x32, u8x32> interleave(
    std::integral_constant<size_t, 32>, u8x32 x0, u8x32 x1) {
  return interleave(std::integral_constant<size_t, 128>{},
                    u8x32{_mm256_unpacklo_epi32(x0.v, x1.v)},
                    u8x32{_mm256_unpackhi_epi32(x0.v, x1.v)});
}
YNN_ALWAYS_INLINE std::tuple<u8x32, u8x32> interleave(
    std::integral_constant<size_t, 16>, u8x32 x0, u8x32 x1) {
  return interleave(std::integral_constant<size_t, 128>{},
                    u8x32{_mm256_unpacklo_epi16(x0.v, x1.v)},
                    u8x32{_mm256_unpackhi_epi16(x0.v, x1.v)});
}
YNN_ALWAYS_INLINE std::tuple<u8x32, u8x32> interleave(
    std::integral_constant<size_t, 8>, u8x32 x0, u8x32 x1) {
  return interleave(std::integral_constant<size_t, 128>{},
                    u8x32{_mm256_unpacklo_epi8(x0.v, x1.v)},
                    u8x32{_mm256_unpackhi_epi8(x0.v, x1.v)});
}
YNN_ALWAYS_INLINE std::tuple<u8x32, u8x32> interleave(
    std::integral_constant<size_t, 4>, u8x32 x0, u8x32 x1) {
  __m256i even0 = _mm256_and_si256(x0.v, _mm256_set1_epi8(0x0f));
  __m256i even1 = _mm256_and_si256(x1.v, _mm256_set1_epi8(0x0f));
  __m256i odd0 = _mm256_and_si256(x0.v, _mm256_set1_epi8(0xf0));
  __m256i odd1 = _mm256_and_si256(x1.v, _mm256_set1_epi8(0xf0));
  return interleave(std::integral_constant<size_t, 8>{},
                    u8x32{_mm256_or_si256(_mm256_slli_epi16(even1, 4), even0)},
                    u8x32{_mm256_or_si256(odd1, _mm256_srli_epi16(odd0, 4))});
}
YNN_ALWAYS_INLINE std::tuple<u8x32, u8x32> interleave(
    std::integral_constant<size_t, 2>, u8x32 x0, u8x32 x1) {
  __m256i even0 = _mm256_and_si256(x0.v, _mm256_set1_epi8(0x33));
  __m256i even1 = _mm256_and_si256(x1.v, _mm256_set1_epi8(0x33));
  __m256i odd0 = _mm256_and_si256(x0.v, _mm256_set1_epi8(0xcc));
  __m256i odd1 = _mm256_and_si256(x1.v, _mm256_set1_epi8(0xcc));
  return interleave(std::integral_constant<size_t, 4>{},
                    u8x32{_mm256_or_si256(_mm256_slli_epi16(even1, 2), even0)},
                    u8x32{_mm256_or_si256(odd1, _mm256_srli_epi16(odd0, 2))});
}
#endif  // YNN_ARCH_X86_AVX2

#ifndef YNN_ARCH_X86_AVX512
// AVX512 provides better versions of these casts.

using f32x16 = vec<float, 16>;
using s32x16 = vec<int32_t, 16>;
using s32x32 = vec<int32_t, 32>;
using f32x32 = vec<float, 32>;
using s16x32 = vec<int16_t, 32>;

YNN_ALWAYS_INLINE s32x16 cast(s8x16 a, int32_t) {
  return {
      s32x8{_mm256_cvtepi8_epi32(a.v)},
      s32x8{_mm256_cvtepi8_epi32(_mm_srli_si128(a.v, 8))},
  };
}

YNN_ALWAYS_INLINE s32x16 cast(u8x16 a, int32_t) {
  return {
      s32x8{_mm256_cvtepu8_epi32(a.v)},
      s32x8{_mm256_cvtepu8_epi32(_mm_srli_si128(a.v, 8))},
  };
}

YNN_ALWAYS_INLINE bf16x16 cast(f32x16 a, bfloat16) {
  __m256 nan_mask_lo = _mm256_cmp_ps(lo(a).v, lo(a).v, _CMP_UNORD_Q);
  __m256i u_lo = _mm256_castps_si256(lo(a).v);
  __m256i lsb_lo =
      _mm256_and_si256(_mm256_srli_epi32(u_lo, 16), _mm256_set1_epi32(1));
  __m256i bias_lo = _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), lsb_lo);
  __m256i res_lo = _mm256_castps_si256(_mm256_blendv_ps(
      _mm256_castsi256_ps(_mm256_add_epi32(u_lo, bias_lo)),
      _mm256_castsi256_ps(_mm256_or_si256(u_lo, _mm256_set1_epi32(0x00010000))),
      nan_mask_lo));
  __m256i c1 = _mm256_srli_epi32(res_lo, 16);

  __m256 nan_mask_hi = _mm256_cmp_ps(hi(a).v, hi(a).v, _CMP_UNORD_Q);
  __m256i u_hi = _mm256_castps_si256(hi(a).v);
  __m256i lsb_hi =
      _mm256_and_si256(_mm256_srli_epi32(u_hi, 16), _mm256_set1_epi32(1));
  __m256i bias_hi = _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), lsb_hi);
  __m256i res_hi = _mm256_castps_si256(_mm256_blendv_ps(
      _mm256_castsi256_ps(_mm256_add_epi32(u_hi, bias_hi)),
      _mm256_castsi256_ps(_mm256_or_si256(u_hi, _mm256_set1_epi32(0x00010000))),
      nan_mask_hi));
  __m256i c2 = _mm256_srli_epi32(res_hi, 16);

  const __m256i d = _mm256_packus_epi32(c1, c2);
  return bf16x16{_mm256_permute4x64_epi64(d, _MM_SHUFFLE(3, 1, 2, 0))};
}

YNN_ALWAYS_INLINE s16x16 cast(s32x16 a, int16_t) {
  const __m256i r = _mm256_packs_epi32(lo(a).v, hi(a).v);
  return s16x16{_mm256_permute4x64_epi64(r, _MM_SHUFFLE(3, 1, 2, 0))};
}

YNN_ALWAYS_INLINE s8x32 cast(s16x32 a, int8_t) {
  const __m256i r = _mm256_packs_epi16(lo(a).v, hi(a).v);
  return s8x32{_mm256_permute4x64_epi64(r, _MM_SHUFFLE(3, 1, 2, 0))};
}

YNN_ALWAYS_INLINE u8x32 cast(s16x32 a, uint8_t) {
  const __m256i r = _mm256_packus_epi16(lo(a).v, hi(a).v);
  return u8x32{_mm256_permute4x64_epi64(r, _MM_SHUFFLE(3, 1, 2, 0))};
}

YNN_ALWAYS_INLINE s16x16 cast(f32x16 f, int16_t) {
  const s32x8 i0 = cast(lo(f), int32_t());
  const s32x8 i1 = cast(hi(f), int32_t());
  return cast(s32x16(i0, i1), int16_t());
}

YNN_ALWAYS_INLINE s8x32 cast(f32x32 f, int8_t) {
  const s16x16 i01 = cast(lo(f), int16_t());
  const s16x16 i23 = cast(hi(f), int16_t());
  return cast(s16x32(i01, i23), int8_t());
}

YNN_ALWAYS_INLINE u8x32 cast(f32x32 f, uint8_t) {
  const s32x8 i0 = cast(lo(lo(f)), int32_t());
  const s32x8 i1 = cast(hi(lo(f)), int32_t());
  const s32x8 i2 = cast(lo(hi(f)), int32_t());
  const s32x8 i3 = cast(hi(hi(f)), int32_t());
  const __m256i i01_16 = _mm256_packs_epi32(i0.v, i1.v);
  const __m256i i23_16 = _mm256_packs_epi32(i2.v, i3.v);
  const __m256i r = _mm256_packus_epi16(i01_16, i23_16);
  return u8x32{_mm256_permutevar8x32_epi32(
      r, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7))};
}
#endif  // YNN_ARCH_X86_AVX512

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_VEC256_BASE_H_
