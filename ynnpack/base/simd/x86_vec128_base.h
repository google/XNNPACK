// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_VEC128_BASE_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_VEC128_BASE_H_

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

namespace ynn {

namespace simd {

// See vec.h for architecture independent comments.

#ifdef YNN_ARCH_X86_AVX512

YNN_ALWAYS_INLINE __mmask8 lo(__mmask16 x) { return (__mmask8)x; }
YNN_ALWAYS_INLINE __mmask8 hi(__mmask16 x) { return (__mmask8)(x >> 8); }
YNN_ALWAYS_INLINE __mmask16 lo(__mmask32 x) { return (__mmask16)x; }
YNN_ALWAYS_INLINE __mmask16 hi(__mmask32 x) { return (__mmask16)(x >> 16); }
YNN_ALWAYS_INLINE __mmask32 lo(__mmask64 x) { return (__mmask32)x; }
YNN_ALWAYS_INLINE __mmask32 hi(__mmask64 x) { return (__mmask32)(x >> 32); }

YNN_ALWAYS_INLINE __mmask16 concat(__mmask8 lo, __mmask8 hi) {
  return ((__mmask16)hi << 8) | lo;
}
YNN_ALWAYS_INLINE __mmask32 concat(__mmask16 lo, __mmask16 hi) {
  return ((__mmask32)hi << 16) | lo;
}
YNN_ALWAYS_INLINE __mmask64 concat(__mmask32 lo, __mmask32 hi) {
  return ((__mmask64)hi << 32) | lo;
}

#endif  // YNN_ARCH_X86_AVX512

template <>
struct vec<float, 4> {
  using value_type = float;
  static constexpr std::integral_constant<size_t, 4> N = {};

  vec() = default;
  explicit vec(__m128 v) : v(v) {}
  vec(float x) : v(_mm_set1_ps(x)) {}  // NOLINT

  __m128 v;
};

template <>
struct vec<bfloat16, 8> {
  using value_type = bfloat16;
  static constexpr std::integral_constant<size_t, 8> N = {};

  vec() = default;
  explicit vec(__m128i v) : v(v) {}
  vec(bfloat16 x) : v(_mm_set1_epi16(x.to_bits())) {}  // NOLINT

  __m128i v;
};

template <>
struct vec<half, 8> {
  using value_type = half;
  static constexpr std::integral_constant<size_t, 8> N = {};

  vec() = default;
  explicit vec(__m128i v) : v(v) {}
  vec(half x) : v(_mm_set1_epi16(x.to_bits())) {}  // NOLINT

  __m128i v;
};

template <>
struct vec<uint16_t, 8> {
  using value_type = uint16_t;
  static constexpr std::integral_constant<size_t, 8> N = {};

  vec() = default;
  explicit vec(__m128i v) : v(v) {}
  vec(uint16_t x) : v(_mm_set1_epi16(x)) {}  // NOLINT

  __m128i v;
};

template <>
struct vec<int16_t, 8> {
  using value_type = int16_t;
  static constexpr std::integral_constant<size_t, 8> N = {};

  vec() = default;
  explicit vec(__m128i v) : v(v) {}
  vec(int16_t x) : v(_mm_set1_epi16(x)) {}  // NOLINT

  __m128i v;
};

template <>
struct vec<uint32_t, 4> {
  using value_type = uint32_t;
  static constexpr std::integral_constant<size_t, 4> N = {};

  vec() = default;
  explicit vec(__m128i v) : v(v) {}
  vec(uint32_t x) : v(_mm_set1_epi32(x)) {}  // NOLINT

  __m128i v;
};

template <>
struct vec<int32_t, 4> {
  using value_type = int32_t;
  static constexpr std::integral_constant<size_t, 4> N = {};

  vec() = default;
  explicit vec(__m128i v) : v(v) {}
  vec(int32_t x) : v(_mm_set1_epi32(x)) {}  // NOLINT

  __m128i v;
};

template <>
struct vec<uint8_t, 16> {
  using value_type = uint8_t;
  static constexpr std::integral_constant<size_t, 16> N = {};

  vec() = default;
  explicit vec(__m128i v) : v(v) {}
  vec(uint8_t x) : v(_mm_set1_epi8(x)) {}  // NOLINT

  __m128i v;
};

template <>
struct vec<int8_t, 16> {
  using value_type = int8_t;
  static constexpr std::integral_constant<size_t, 16> N = {};

  vec() = default;
  explicit vec(__m128i v) : v(v) {}
  vec(int8_t x) : v(_mm_set1_epi8(x)) {}  // NOLINT

  __m128i v;
};

template <>
struct vec<int64_t, 2> {
  using value_type = int64_t;
  static constexpr std::integral_constant<size_t, 2> N = {};

  vec() = default;
  explicit vec(__m128i v) : v(v) {}
  vec(int64_t x) : v(_mm_set1_epi64x(x)) {}  // NOLINT

  __m128i v;
};

struct s2x64 {
  __m128i v;
};

struct s2x32 {
  uint64_t v;
};

struct s4x32 {
  __m128i v;
};

template <>
struct vec<double, 2> {
  using value_type = double;
  static constexpr std::integral_constant<size_t, 2> N = {};

  vec() = default;
  explicit vec(__m128d v) : v(v) {}
  vec(vec<double, 1> lo, vec<double, 1> hi) : v(_mm_set_pd(hi.v, lo.v)) {}
  vec(double x) : v(_mm_set1_pd(x)) {}  // NOLINT

  __m128d v;
};

using s64x2 = vec<int64_t, 2>;
using f64x2 = vec<double, 2>;
using f32x4 = vec<float, 4>;
using u32x4 = vec<uint32_t, 4>;
using s32x4 = vec<int32_t, 4>;
using bf16x8 = vec<bfloat16, 8>;
using f16x8 = vec<half, 8>;
using u16x8 = vec<uint16_t, 8>;
using s16x8 = vec<int16_t, 8>;
using u8x16 = vec<uint8_t, 16>;
using s8x16 = vec<int8_t, 16>;

YNN_ALWAYS_INLINE vec<double, 1> lo(f64x2 x) {
  return vec<double, 1>{_mm_cvtsd_f64(x.v)};
}
YNN_ALWAYS_INLINE vec<double, 1> hi(f64x2 x) {
  return vec<double, 1>{_mm_cvtsd_f64(_mm_unpackhi_pd(x.v, x.v))};
}

YNN_ALWAYS_INLINE vec<int64_t, 1> lo(s64x2 x) {
  return vec<int64_t, 1>{_mm_cvtsi128_si64(x.v)};
}
YNN_ALWAYS_INLINE vec<int64_t, 1> hi(s64x2 x) {
  return vec<int64_t, 1>{_mm_cvtsi128_si64(_mm_unpackhi_epi64(x.v, x.v))};
}

namespace internal {

// These overloads are x86-specific helpers for implementing templated
// interleave/transpose helpers that are not x86-specific.
YNN_ALWAYS_INLINE __m128 movehl(__m128 a, __m128 b) {
  return _mm_movehl_ps(a, b);
}
YNN_ALWAYS_INLINE __m128 movelh(__m128 a, __m128 b) {
  return _mm_movelh_ps(a, b);
}

YNN_ALWAYS_INLINE __m128i movehl(__m128i a, __m128i b) {
  return _mm_castps_si128(
      _mm_movehl_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b)));
}
YNN_ALWAYS_INLINE __m128i movelh(__m128i a, __m128i b) {
  return _mm_castps_si128(
      _mm_movelh_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b)));
}

}  // namespace internal

YNN_ALWAYS_INLINE f64x2 load_aligned(const double* ptr,
                                     decltype(f64x2::N), f64x2 = {}) {
  return f64x2{_mm_load_pd(ptr)};
}
YNN_ALWAYS_INLINE f32x4 load_aligned(const float* ptr, decltype(f32x4::N),
                                     f32x4 = {}) {
  return f32x4{_mm_load_ps(ptr)};
}
YNN_ALWAYS_INLINE s32x4 load_aligned(const int32_t* ptr, decltype(s32x4::N),
                                     s32x4 = {}) {
  return s32x4{_mm_load_si128(reinterpret_cast<const __m128i*>(ptr))};
}
YNN_ALWAYS_INLINE bf16x8 load_aligned(const bfloat16* ptr, decltype(bf16x8::N),
                                      bf16x8 = {}) {
  return bf16x8{_mm_load_si128(reinterpret_cast<const __m128i*>(ptr))};
}
YNN_ALWAYS_INLINE f16x8 load_aligned(const half* ptr, decltype(f16x8::N),
                                     f16x8 = {}) {
  return f16x8{_mm_load_si128(reinterpret_cast<const __m128i*>(ptr))};
}
YNN_ALWAYS_INLINE s16x8 load_aligned(const int16_t* ptr, decltype(s16x8::N),
                                     s16x8 = {}) {
  return s16x8{_mm_load_si128(reinterpret_cast<const __m128i*>(ptr))};
}
YNN_ALWAYS_INLINE u8x16 load_aligned(const uint8_t* ptr, decltype(u8x16::N),
                                     u8x16 = {}) {
  return u8x16{_mm_load_si128(reinterpret_cast<const __m128i*>(ptr))};
}
YNN_ALWAYS_INLINE s8x16 load_aligned(const int8_t* ptr, decltype(s8x16::N),
                                     s8x16 = {}) {
  return s8x16{_mm_load_si128(reinterpret_cast<const __m128i*>(ptr))};
}

YNN_ALWAYS_INLINE void store_aligned(double* ptr, f64x2 b,
                                     decltype(f64x2::N) = {}) {
  _mm_store_pd(ptr, b.v);
}
YNN_ALWAYS_INLINE void store_aligned(float* ptr, f32x4 b,
                                     decltype(f32x4::N) = {}) {
  _mm_store_ps(ptr, b.v);
}
YNN_ALWAYS_INLINE void store_aligned(bfloat16* ptr, bf16x8 b,
                                     decltype(bf16x8::N) = {}) {
  _mm_store_si128(reinterpret_cast<__m128i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store_aligned(half* ptr, f16x8 b,
                                     decltype(f16x8::N) = {}) {
  _mm_store_si128(reinterpret_cast<__m128i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store_aligned(uint16_t* ptr, u16x8 b,
                                     decltype(u16x8::N) = {}) {
  _mm_store_si128(reinterpret_cast<__m128i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store_aligned(int16_t* ptr, s16x8 b,
                                     decltype(s16x8::N) = {}) {
  _mm_store_si128(reinterpret_cast<__m128i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store_aligned(uint32_t* ptr, u32x4 b,
                                     decltype(u32x4::N) = {}) {
  _mm_store_si128(reinterpret_cast<__m128i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store_aligned(int32_t* ptr, s32x4 b,
                                     decltype(s32x4::N) = {}) {
  _mm_store_si128(reinterpret_cast<__m128i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store_aligned(uint8_t* ptr, u8x16 b,
                                     decltype(u8x16::N) = {}) {
  _mm_store_si128(reinterpret_cast<__m128i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store_aligned(int8_t* ptr, s8x16 b,
                                     decltype(s8x16::N) = {}) {
  _mm_store_si128(reinterpret_cast<__m128i*>(ptr), b.v);
}

YNN_ALWAYS_INLINE f64x2 load(const double* ptr,
                             decltype(f64x2::N), f64x2 = {}) {
  return f64x2{_mm_loadu_pd(ptr)};
}
YNN_ALWAYS_INLINE f32x4 load(const float* ptr, decltype(f32x4::N), f32x4 = {}) {
  return f32x4{_mm_loadu_ps(ptr)};
}
YNN_ALWAYS_INLINE s32x4 load(const int32_t* ptr, decltype(s32x4::N),
                             s32x4 = {}) {
  return s32x4{_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr))};
}
YNN_ALWAYS_INLINE bf16x8 load(const bfloat16* ptr, decltype(bf16x8::N),
                              bf16x8 = {}) {
  return bf16x8{_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr))};
}
YNN_ALWAYS_INLINE f16x8 load(const half* ptr, decltype(f16x8::N), f16x8 = {}) {
  return f16x8{_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr))};
}
YNN_ALWAYS_INLINE u16x8 load(const uint16_t* ptr, decltype(u16x8::N),
                             u16x8 = {}) {
  return u16x8{_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr))};
}
YNN_ALWAYS_INLINE s16x8 load(const int16_t* ptr, decltype(s16x8::N),
                             s16x8 = {}) {
  return s16x8{_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr))};
}
YNN_ALWAYS_INLINE u8x16 load(const uint8_t* ptr, decltype(u8x16::N),
                             u8x16 = {}) {
  return u8x16{_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr))};
}
YNN_ALWAYS_INLINE s8x16 load(const int8_t* ptr, decltype(s8x16::N),
                             s8x16 = {}) {
  return s8x16{_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr))};
}

YNN_ALWAYS_INLINE void store(double* ptr, f64x2 b, decltype(f64x2::N) = {}) {
  _mm_storeu_pd(ptr, b.v);
}
YNN_ALWAYS_INLINE void store(float* ptr, f32x4 b, decltype(f32x4::N) = {}) {
  _mm_storeu_ps(ptr, b.v);
}
YNN_ALWAYS_INLINE void store(uint32_t* ptr, u32x4 b, decltype(u32x4::N) = {}) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x4 b, decltype(s32x4::N) = {}) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(bfloat16* ptr, bf16x8 b,
                             decltype(bf16x8::N) = {}) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(half* ptr, f16x8 b, decltype(f16x8::N) = {}) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(uint16_t* ptr, u16x8 b, decltype(u16x8::N) = {}) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(int16_t* ptr, s16x8 b, decltype(s16x8::N) = {}) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x16 b, decltype(u8x16::N) = {}) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(int8_t* ptr, s8x16 b, decltype(s8x16::N) = {}) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), b.v);
}

#ifdef YNN_ARCH_X86_AVX512

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

#else

namespace internal {

// This implementation of std::copy_n takes a constant upper bound for the
// number of elements.
template <typename T>
YNN_ALWAYS_INLINE void copy_n_small(const T* src, size_t n, T* dst,
                                    std::integral_constant<size_t, 16>) {
  assert(n <= 16);
  if (n == 16) {
    memcpy(dst, src, 16 * sizeof(T));
    return;
  }
  if (n & 8) {
    memcpy(dst, src, 8 * sizeof(T));
    dst += 8;
    src += 8;
  }
  if (n & 4) {
    memcpy(dst, src, 4 * sizeof(T));
    dst += 4;
    src += 4;
  }
  if (n & 2) {
    memcpy(dst, src, 2 * sizeof(T));
    dst += 2;
    src += 2;
  }
  if (n & 1) {
    memcpy(dst, src, 1 * sizeof(T));
  }
}

template <typename T>
YNN_ALWAYS_INLINE void copy_n_small(const T* src, size_t n, T* dst,
                                    std::integral_constant<size_t, 8>) {
  assert(n <= 8);
  if (n == 8) {
    memcpy(dst, src, 8 * sizeof(T));
    return;
  }
  if (n & 4) {
    memcpy(dst, src, 4 * sizeof(T));
    dst += 4;
    src += 4;
  }
  if (n & 2) {
    memcpy(dst, src, 2 * sizeof(T));
    dst += 2;
    src += 2;
  }
  if (n & 1) {
    memcpy(dst, src, 1 * sizeof(T));
  }
}

template <typename T>
YNN_ALWAYS_INLINE void copy_n_small(const T* src, size_t n, T* dst,
                                    std::integral_constant<size_t, 4>) {
  assert(n <= 4);
  switch (n) {
    // clang-format off
    case 4: dst[3] = src[3]; [[fallthrough]];
    case 3: dst[2] = src[2]; [[fallthrough]];
    case 2: dst[1] = src[1]; [[fallthrough]];
    case 1: dst[0] = src[0];
    // clang-format on
  }
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE void store_aligned(T* dst, zeros<N>) {
  memset(dst, 0, N * sizeof(T));
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE void store_aligned(T* dst, undef<N>) {}

// This partial load implements the simplest strategy:
// 1. Copy the src value to memory
// 2. Do a small memcpy from the pointer to load from.
// 3. Load the memory to a vector.
template <typename T, typename Init>
vec<T, Init::N> partial_load_sse(const T* ptr, size_t n, Init src) {
  alignas(sizeof(vec<T, Init::N>)) T lanes[Init::N];
  store_aligned(lanes, src);
  copy_n_small(ptr, n, lanes, Init::N);
  return load_aligned(lanes, Init::N);
}

// Load 64-bits from ptr into the low 64-bits of `v`.
YNN_ALWAYS_INLINE __m128i load_64(__m128i v, const void* ptr) {
  return _mm_castps_si128(
      _mm_loadl_pi(_mm_castsi128_ps(v), reinterpret_cast<const __m64*>(ptr)));
}
YNN_ALWAYS_INLINE __m128 load_64(__m128 v, const void* ptr) {
  return _mm_loadl_pi(v, reinterpret_cast<const __m64*>(ptr));
}

// Load 32-bits from `ptr` into a zero vector.
YNN_ALWAYS_INLINE __m128i load_32_zero(const int32_t* ptr) {
  return _mm_castps_si128(_mm_load_ss(reinterpret_cast<const float*>(ptr)));
}
YNN_ALWAYS_INLINE __m128 load_32_zero(const float* ptr) {
  return _mm_load_ss(ptr);
}

// Uses various SSE load instructions to implement partial loads of 32-bits at a
// time.
template <typename T>
vec<T, 4> partial_load_sse(const T* ptr, size_t n, zeros<4>) {
  assert(n <= 4);
  if (n == 4) {
    return load(ptr, std::integral_constant<size_t, 4>{});
  }
  vec<T, 4> result(T{0});
  switch (n) {
    case 3:
      return vec<T, 4>{movelh(load_64(result.v, ptr), load_32_zero(ptr + 2))};
    case 2:
      return vec<T, 4>{load_64(result.v, ptr)};
    case 1:
      return vec<T, 4>{load_32_zero(ptr)};
  }
  return result;
}

template <typename T>
YNN_ALWAYS_INLINE vec<T, 4> partial_load_sse(const T* ptr, size_t n, undef<4>) {
  return partial_load_sse(ptr, n, zeros<4>{});
}

// Store the low 64-bits of `v` to `ptr`.
YNN_ALWAYS_INLINE void store_64(void* ptr, __m128i v) {
  _mm_storel_pi(reinterpret_cast<__m64*>(ptr), _mm_castsi128_ps(v));
}
YNN_ALWAYS_INLINE void store_64(void* ptr, __m128 v) {
  _mm_storel_pi(reinterpret_cast<__m64*>(ptr), v);
}

// Store the low 32-bits of `v` to `ptr`.
YNN_ALWAYS_INLINE void store_32(void* ptr, __m128i v) {
  _mm_store_ss(reinterpret_cast<float*>(ptr), _mm_castsi128_ps(v));
}
YNN_ALWAYS_INLINE void store_32(void* ptr, __m128 v) {
  _mm_store_ss(reinterpret_cast<float*>(ptr), v);
}

// Use various SSE instructions to do 32- or 64-bit stores.
template <typename T>
void partial_store_sse(T* ptr, vec<T, 4> b, size_t n) {
  assert(n <= 4);
  if (n == 4) {
    store(ptr, b, std::integral_constant<size_t, 4>{});
    return;
  }
  if (n & 2) {
    store_64(ptr, b.v);
    ptr += 2;
    b.v = movehl(b.v, b.v);
  }
  if (n & 1) {
    store_32(ptr, b.v);
  }
}

template <typename T>
void partial_store_sse(T* ptr, vec<T, 8> b, size_t n) {
  assert(n <= 8);
  if (n == 8) {
    store(ptr, b, std::integral_constant<size_t, 8>{});
    return;
  }
  if (n & 4) {
    store_64(ptr, b.v);
    ptr += 4;
    b.v = movehl(b.v, b.v);
  }
  if (n & 2) {
    store_32(ptr, b.v);
    ptr += 2;
    b.v = _mm_srli_si128(b.v, 4);
  }
  // We might need a fixup of sub-32-bit values.
  T x[2];
  store_32(x, b.v);
  if (n & 1) {
    *ptr = x[0];
  }
}

template <typename T>
void partial_store_sse(T* ptr, vec<T, 16> b, size_t n) {
  assert(n <= 16);
  if (n == 16) {
    store(ptr, b, std::integral_constant<size_t, 16>{});
    return;
  }
  if (n & 8) {
    store_64(ptr, b.v);
    ptr += 8;
    b.v = movehl(b.v, b.v);
  }
  if (n & 4) {
    store_32(ptr, b.v);
    ptr += 4;
    b.v = _mm_srli_si128(b.v, 4);
  }
  // We might need a fixup of sub-32-bit values.
  T x[4];
  store_32(x, b.v);
  switch (n & 3) {
      // clang-format off
    case 3: ptr[2] = x[2]; [[fallthrough]];
    case 2: ptr[1] = x[1]; [[fallthrough]];
    case 1: ptr[0] = x[0];
      // clang-format on
  }
}

}  // namespace internal

YNN_ALWAYS_INLINE f32x4 load(const float* ptr, size_t n, f32x4 src) {
  return internal::partial_load_sse(ptr, n, src);
}
YNN_ALWAYS_INLINE s32x4 load(const int32_t* ptr, size_t n, s32x4 src) {
  return internal::partial_load_sse(ptr, n, src);
}
YNN_ALWAYS_INLINE bf16x8 load(const bfloat16* ptr, size_t n, bf16x8 src) {
  return internal::partial_load_sse(ptr, n, src);
}
YNN_ALWAYS_INLINE f16x8 load(const half* ptr, size_t n, f16x8 src) {
  return internal::partial_load_sse(ptr, n, src);
}
YNN_ALWAYS_INLINE s16x8 load(const int16_t* ptr, size_t n, s16x8 src) {
  return internal::partial_load_sse(ptr, n, src);
}
YNN_ALWAYS_INLINE u8x16 load(const uint8_t* ptr, size_t n, u8x16 src) {
  return internal::partial_load_sse(ptr, n, src);
}
YNN_ALWAYS_INLINE s8x16 load(const int8_t* ptr, size_t n, s8x16 src) {
  return internal::partial_load_sse(ptr, n, src);
}

YNN_ALWAYS_INLINE f32x4 load(const float* ptr, size_t n, zeros<4> src) {
  return internal::partial_load_sse(ptr, n, src);
}
YNN_ALWAYS_INLINE s32x4 load(const int32_t* ptr, size_t n, zeros<4> src) {
  return internal::partial_load_sse(ptr, n, src);
}
YNN_ALWAYS_INLINE bf16x8 load(const bfloat16* ptr, size_t n, zeros<8> src) {
  return internal::partial_load_sse(ptr, n, src);
}
YNN_ALWAYS_INLINE f16x8 load(const half* ptr, size_t n, zeros<8> src) {
  return internal::partial_load_sse(ptr, n, src);
}
YNN_ALWAYS_INLINE s16x8 load(const int16_t* ptr, size_t n, zeros<8> src) {
  return internal::partial_load_sse(ptr, n, src);
}
YNN_ALWAYS_INLINE u8x16 load(const uint8_t* ptr, size_t n, zeros<16> src) {
  return internal::partial_load_sse(ptr, n, src);
}
YNN_ALWAYS_INLINE s8x16 load(const int8_t* ptr, size_t n, zeros<16> src) {
  return internal::partial_load_sse(ptr, n, src);
}

YNN_ALWAYS_INLINE f32x4 load(const float* ptr, size_t n, undef<4> src) {
  return internal::partial_load_sse(ptr, n, src);
}
YNN_ALWAYS_INLINE s32x4 load(const int32_t* ptr, size_t n, undef<4> src) {
  return internal::partial_load_sse(ptr, n, src);
}
YNN_ALWAYS_INLINE bf16x8 load(const bfloat16* ptr, size_t n, undef<8> src) {
  return internal::partial_load_sse(ptr, n, src);
}
YNN_ALWAYS_INLINE f16x8 load(const half* ptr, size_t n, undef<8> src) {
  return internal::partial_load_sse(ptr, n, src);
}
YNN_ALWAYS_INLINE s16x8 load(const int16_t* ptr, size_t n, undef<8> src) {
  return internal::partial_load_sse(ptr, n, src);
}
YNN_ALWAYS_INLINE u8x16 load(const uint8_t* ptr, size_t n, undef<16> src) {
  return internal::partial_load_sse(ptr, n, src);
}
YNN_ALWAYS_INLINE s8x16 load(const int8_t* ptr, size_t n, undef<16> src) {
  return internal::partial_load_sse(ptr, n, src);
}

YNN_ALWAYS_INLINE void store(float* ptr, f32x4 b, size_t n) {
  internal::partial_store_sse(ptr, b, n);
}
YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x4 b, size_t n) {
  internal::partial_store_sse(ptr, b, n);
}
YNN_ALWAYS_INLINE void store(bfloat16* ptr, bf16x8 b, size_t n) {
  internal::partial_store_sse(ptr, b, n);
}
YNN_ALWAYS_INLINE void store(half* ptr, f16x8 b, size_t n) {
  internal::partial_store_sse(ptr, b, n);
}
YNN_ALWAYS_INLINE void store(int16_t* ptr, s16x8 b, size_t n) {
  internal::partial_store_sse(ptr, b, n);
}
YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x16 val, size_t n) {
  internal::partial_store_sse(ptr, val, n);
}
YNN_ALWAYS_INLINE void store(int8_t* ptr, s8x16 val, size_t n) {
  internal::partial_store_sse(ptr, val, n);
}

#endif

YNN_ALWAYS_INLINE f64x2 operator+(f64x2 a, f64x2 b) {
  return f64x2{_mm_add_pd(a.v, b.v)};
}
YNN_ALWAYS_INLINE f32x4 operator+(f32x4 a, f32x4 b) {
  return f32x4{_mm_add_ps(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x4 operator+(s32x4 a, s32x4 b) {
  return s32x4{_mm_add_epi32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 operator+(s16x8 a, s16x8 b) {
  return s16x8{_mm_add_epi16(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 operator+(s8x16 a, s8x16 b) {
  return s8x16{_mm_add_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 operator+(u8x16 a, u8x16 b) {
  return u8x16{_mm_add_epi8(a.v, b.v)};
}

YNN_ALWAYS_INLINE f64x2 operator-(f64x2 a, f64x2 b) {
  return f64x2{_mm_sub_pd(a.v, b.v)};
}
YNN_ALWAYS_INLINE f32x4 operator-(f32x4 a, f32x4 b) {
  return f32x4{_mm_sub_ps(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x4 operator-(s32x4 a, s32x4 b) {
  return s32x4{_mm_sub_epi32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 operator-(s16x8 a, s16x8 b) {
  return s16x8{_mm_sub_epi16(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 operator-(s8x16 a, s8x16 b) {
  return s8x16{_mm_sub_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 operator-(u8x16 a, u8x16 b) {
  return u8x16{_mm_sub_epi8(a.v, b.v)};
}

YNN_ALWAYS_INLINE f64x2 operator-(f64x2 a) {
  return f64x2{_mm_xor_pd(_mm_set1_pd(-0.0f), a.v)};
}
YNN_ALWAYS_INLINE f32x4 operator-(f32x4 a) {
  return f32x4{_mm_xor_ps(_mm_set1_ps(-0.0f), a.v)};
}

#ifdef YNN_ARCH_X86_SSE41
YNN_ALWAYS_INLINE s32x4& operator*=(s32x4& a, s32x4 b) {
  a.v = _mm_mullo_epi32(a.v, b.v);
  return a;
}

YNN_ALWAYS_INLINE s32x4 operator*(s32x4 a, s32x4 b) { return a *= b; }
#endif  // YNN_ARCH_X86_SSE41

YNN_ALWAYS_INLINE s16x8 add_sat(s16x8 a, s16x8 b) {
  return s16x8{_mm_adds_epi16(a.v, b.v)};
}
YNN_ALWAYS_INLINE u16x8 add_sat(u16x8 a, u16x8 b) {
  return u16x8{_mm_adds_epu16(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 add_sat(s8x16 a, s8x16 b) {
  return s8x16{_mm_adds_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 add_sat(u8x16 a, u8x16 b) {
  return u8x16{_mm_adds_epu8(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 sub_sat(s16x8 a, s16x8 b) {
  return s16x8{_mm_subs_epi16(a.v, b.v)};
}
YNN_ALWAYS_INLINE u16x8 sub_sat(u16x8 a, u16x8 b) {
  return u16x8{_mm_subs_epu16(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 sub_sat(s8x16 a, s8x16 b) {
  return s8x16{_mm_subs_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 sub_sat(u8x16 a, u8x16 b) {
  return u8x16{_mm_subs_epu8(a.v, b.v)};
}

YNN_ALWAYS_INLINE f64x2 operator*(f64x2 a, f64x2 b) {
  return f64x2{_mm_mul_pd(a.v, b.v)};
}
YNN_ALWAYS_INLINE f32x4 operator*(f32x4 a, f32x4 b) {
  return f32x4{_mm_mul_ps(a.v, b.v)};
}
YNN_ALWAYS_INLINE f32x4 operator/(f32x4 a, f32x4 b) {
  return f32x4{_mm_div_ps(a.v, b.v)};
}
YNN_ALWAYS_INLINE f64x2 operator/(f64x2 a, f64x2 b) {
  return f64x2{_mm_div_pd(a.v, b.v)};
}

#ifdef YNN_ARCH_FP16_ARITHMETIC
YNN_ALWAYS_INLINE f16x8 operator+(f16x8 a, f16x8 b) {
  return f16x8{_mm_castph_si128(
      _mm_add_ph(_mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v)))};
}
YNN_ALWAYS_INLINE f16x8 operator-(f16x8 a, f16x8 b) {
  return f16x8{_mm_castph_si128(
      _mm_sub_ph(_mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v)))};
}
YNN_ALWAYS_INLINE f16x8 operator*(f16x8 a, f16x8 b) {
  return f16x8{_mm_castph_si128(
      _mm_mul_ph(_mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v)))};
}
YNN_ALWAYS_INLINE f16x8 operator/(f16x8 a, f16x8 b) {
  return f16x8{_mm_castph_si128(
      _mm_div_ph(_mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v)))};
}
YNN_ALWAYS_INLINE f16x8 operator-(f16x8 a) {
  return f16x8{_mm_xor_si128(_mm_set1_epi16(0x8000), a.v)};
}
#endif

YNN_ALWAYS_INLINE s16x8 operator&(s16x8 a, s16x8 b) {
  return s16x8{_mm_and_si128(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 operator>>(s16x8 a, int b) {
  return s16x8{_mm_srai_epi16(a.v, b)};
}
YNN_ALWAYS_INLINE s16x8 operator<<(s16x8 a, int b) {
  return s16x8{_mm_slli_epi16(a.v, b)};
}
YNN_ALWAYS_INLINE s16x8 operator^(s16x8 a, s16x8 b) {
  return s16x8{_mm_xor_si128(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 operator|(s16x8 a, s16x8 b) {
  return s16x8{_mm_or_si128(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 operator~(s16x8 a) {
  return s16x8{_mm_xor_si128(a.v, _mm_set1_epi32(-1))};
}

YNN_ALWAYS_INLINE s32x4 operator&(s32x4 a, s32x4 b) {
  return s32x4{_mm_and_si128(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x4 operator|(s32x4 a, s32x4 b) {
  return s32x4{_mm_or_si128(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x4 operator^(s32x4 a, s32x4 b) {
  return s32x4{_mm_xor_si128(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x4 operator~(s32x4 a) {
  return s32x4{_mm_xor_si128(a.v, _mm_set1_epi32(-1))};
}
YNN_ALWAYS_INLINE s32x4 operator<<(s32x4 a, int b) {
  return s32x4{_mm_slli_epi32(a.v, b)};
}
YNN_ALWAYS_INLINE u8x16 operator&(u8x16 a, u8x16 b) {
  return u8x16{_mm_and_si128(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 operator|(u8x16 a, u8x16 b) {
  return u8x16{_mm_or_si128(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 operator^(u8x16 a, u8x16 b) {
  return u8x16{_mm_xor_si128(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 operator~(u8x16 a) {
  return u8x16{_mm_xor_si128(a.v, _mm_set1_epi32(-1))};
}

YNN_ALWAYS_INLINE s8x16 operator&(s8x16 a, s8x16 b) {
  return s8x16{_mm_and_si128(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 operator|(s8x16 a, s8x16 b) {
  return s8x16{_mm_or_si128(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 operator^(s8x16 a, s8x16 b) {
  return s8x16{_mm_xor_si128(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 operator~(s8x16 a) {
  return s8x16{_mm_xor_si128(a.v, _mm_set1_epi32(-1))};
}

YNN_ALWAYS_INLINE u32x4 operator&(u32x4 a, u32x4 b) {
  return u32x4{_mm_and_si128(a.v, b.v)};
}
YNN_ALWAYS_INLINE u32x4 operator|(u32x4 a, u32x4 b) {
  return u32x4{_mm_or_si128(a.v, b.v)};
}
YNN_ALWAYS_INLINE u32x4 operator^(u32x4 a, u32x4 b) {
  return u32x4{_mm_xor_si128(a.v, b.v)};
}
YNN_ALWAYS_INLINE u32x4 operator~(u32x4 a) {
  return u32x4{_mm_xor_si128(a.v, _mm_set1_epi32(-1))};
}

YNN_ALWAYS_INLINE u16x8 operator&(u16x8 a, u16x8 b) {
  return u16x8{_mm_and_si128(a.v, b.v)};
}
YNN_ALWAYS_INLINE u16x8 operator|(u16x8 a, u16x8 b) {
  return u16x8{_mm_or_si128(a.v, b.v)};
}
YNN_ALWAYS_INLINE u16x8 operator^(u16x8 a, u16x8 b) {
  return u16x8{_mm_xor_si128(a.v, b.v)};
}
YNN_ALWAYS_INLINE u16x8 operator~(u16x8 a) {
  return u16x8{_mm_xor_si128(a.v, _mm_set1_epi32(-1))};
}

YNN_ALWAYS_INLINE s64x2 operator&(s64x2 a, s64x2 b) {
  return s64x2{_mm_and_si128(a.v, b.v)};
}
YNN_ALWAYS_INLINE s64x2 operator|(s64x2 a, s64x2 b) {
  return s64x2{_mm_or_si128(a.v, b.v)};
}
YNN_ALWAYS_INLINE s64x2 operator^(s64x2 a, s64x2 b) {
  return s64x2{_mm_xor_si128(a.v, b.v)};
}
YNN_ALWAYS_INLINE s64x2 operator~(s64x2 a) {
  return s64x2{_mm_xor_si128(a.v, _mm_set1_epi32(-1))};
}

YNN_ALWAYS_INLINE s16x8 operator&(s16x8 a, int b) {
  return a & s16x8((int16_t)b);
}

YNN_ALWAYS_INLINE f32x4 min(f32x4 a, f32x4 b) {
  return f32x4{_mm_min_ps(a.v, b.v)};
}
YNN_ALWAYS_INLINE f64x2 min(f64x2 a, f64x2 b) {
  return f64x2{_mm_min_pd(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 min(s16x8 a, s16x8 b) {
  return s16x8{_mm_min_epi16(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 min(u8x16 a, u8x16 b) {
  return u8x16{_mm_min_epu8(a.v, b.v)};
}

YNN_ALWAYS_INLINE f32x4 max(f32x4 a, f32x4 b) {
  return f32x4{_mm_max_ps(a.v, b.v)};
}
YNN_ALWAYS_INLINE f64x2 max(f64x2 a, f64x2 b) {
  return f64x2{_mm_max_pd(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 max(s16x8 a, s16x8 b) {
  return s16x8{_mm_max_epi16(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 max(u8x16 a, u8x16 b) {
  return u8x16{_mm_max_epu8(a.v, b.v)};
}

// x86 min/max only preserve NaN if it is the second argument. So, rewrite
// min/max with constant RHS to put the constant on the LHS instead. This avoids
// the need for explicit NaN propagation logic.
YNN_ALWAYS_INLINE f32x4 min(f32x4 a, float b) {
  return f32x4{_mm_min_ps(_mm_set1_ps(b), a.v)};
}
YNN_ALWAYS_INLINE f64x2 min(f64x2 a, double b) {
  return f64x2{_mm_min_pd(_mm_set1_pd(b), a.v)};
}
YNN_ALWAYS_INLINE f32x4 max(f32x4 a, float b) {
  return f32x4{_mm_max_ps(_mm_set1_ps(b), a.v)};
}
YNN_ALWAYS_INLINE f64x2 max(f64x2 a, double b) {
  return f64x2{_mm_max_pd(_mm_set1_pd(b), a.v)};
}

#ifdef YNN_ARCH_FP16_ARITHMETIC
YNN_ALWAYS_INLINE f16x8 min(f16x8 a, f16x8 b) {
  return f16x8{_mm_castph_si128(
      _mm_min_ph(_mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v)))};
}
YNN_ALWAYS_INLINE f16x8 max(f16x8 a, f16x8 b) {
  return f16x8{_mm_castph_si128(
      _mm_max_ph(_mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v)))};
}
YNN_ALWAYS_INLINE f16x8 min(f16x8 a, half b) {
  return f16x8{
      _mm_castph_si128(_mm_min_ph(_mm_set1_ph(b), _mm_castsi128_ph(a.v)))};
}
YNN_ALWAYS_INLINE f16x8 max(f16x8 a, half b) {
  return f16x8{
      _mm_castph_si128(_mm_max_ph(_mm_set1_ph(b), _mm_castsi128_ph(a.v)))};
}
#endif

#ifdef YNN_ARCH_X86_SSE41
YNN_ALWAYS_INLINE s8x16 min(s8x16 a, s8x16 b) {
  return s8x16{_mm_min_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 max(s8x16 a, s8x16 b) {
  return s8x16{_mm_max_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x4 min(s32x4 a, s32x4 b) {
  return s32x4{_mm_min_epi32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x4 max(s32x4 a, s32x4 b) {
  return s32x4{_mm_max_epi32(a.v, b.v)};
}
#endif  // YNN_ARCH_X86_SSE41

YNN_ALWAYS_INLINE f32x4 sqrt(f32x4 a) { return f32x4{_mm_sqrt_ps(a.v)}; }
YNN_ALWAYS_INLINE f64x2 sqrt(f64x2 a) { return f64x2{_mm_sqrt_pd(a.v)}; }

YNN_ALWAYS_INLINE f32x4 copysign(f32x4 mag, f32x4 sgn) {
  __m128 sign_mask = _mm_set1_ps(-0.0f);
  return f32x4{
      _mm_or_ps(_mm_and_ps(sign_mask, sgn.v), _mm_andnot_ps(sign_mask, mag.v))};
}
YNN_ALWAYS_INLINE f64x2 copysign(f64x2 mag, f64x2 sgn) {
  __m128d sign_mask = _mm_set1_pd(-0.0);
  return f64x2{
      _mm_or_pd(_mm_and_pd(sign_mask, sgn.v), _mm_andnot_pd(sign_mask, mag.v))};
}

YNN_ALWAYS_INLINE f32x4 abs(f32x4 a) {
  return f32x4{_mm_andnot_ps(_mm_set1_ps(-0.0), a.v)};
}
YNN_ALWAYS_INLINE f64x2 abs(f64x2 a) {
  return f64x2{_mm_andnot_pd(_mm_set1_pd(-0.0), a.v)};
}
#ifdef YNN_ARCH_X86_SSE41
YNN_ALWAYS_INLINE u8x16 abs(s8x16 a) { return u8x16{_mm_abs_epi8(a.v)}; }
YNN_ALWAYS_INLINE u16x8 abs(s16x8 a) { return u16x8{_mm_abs_epi16(a.v)}; }
YNN_ALWAYS_INLINE u32x4 abs(s32x4 a) { return u32x4{_mm_abs_epi32(a.v)}; }

YNN_ALWAYS_INLINE f32x4 floor(f32x4 a) { return f32x4{_mm_floor_ps(a.v)}; }
YNN_ALWAYS_INLINE f64x2 floor(f64x2 a) { return f64x2{_mm_floor_pd(a.v)}; }
YNN_ALWAYS_INLINE f32x4 ceil(f32x4 a) { return f32x4{_mm_ceil_ps(a.v)}; }
YNN_ALWAYS_INLINE f64x2 ceil(f64x2 a) { return f64x2{_mm_ceil_pd(a.v)}; }
YNN_ALWAYS_INLINE f32x4 round(f32x4 a) {
  return f32x4{
      _mm_round_ps(a.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)};
}
YNN_ALWAYS_INLINE f64x2 round(f64x2 a) {
  return f64x2{
      _mm_round_pd(a.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)};
}
#endif  // YNN_ARCH_X86_SSE41

#ifdef YNN_ARCH_X86_AVX512
YNN_ALWAYS_INLINE f32x4 floor_log2(f32x4 a) {
  // getexp handles 0 correctly, but not negative numbers.
  __m128 res = _mm_getexp_ps(a.v);
  __mmask8 negative = _mm_cmp_ps_mask(a.v, _mm_setzero_ps(), _CMP_LT_OQ);
  return f32x4{_mm_mask_blend_ps(
      negative, res, _mm_set1_ps(std::numeric_limits<float>::quiet_NaN()))};
}
YNN_ALWAYS_INLINE f64x2 floor_log2(f64x2 a) {
  // getexp handles 0 correctly, but not negative numbers.
  __m128d res = _mm_getexp_pd(a.v);
  __mmask8 negative = _mm_cmp_pd_mask(a.v, _mm_setzero_pd(), _CMP_LT_OQ);
  return f64x2{_mm_mask_blend_pd(
      negative, res, _mm_set1_pd(std::numeric_limits<double>::quiet_NaN()))};
}

#ifdef YNN_ARCH_FP16_ARITHMETIC
YNN_ALWAYS_INLINE f16x8 floor_log2(f16x8 a) {
  // getexp handles 0 correctly, but not negative numbers.
  __m128h res = _mm_getexp_ph(_mm_castsi128_ph(a.v));
  __mmask8 negative =
      _mm_cmp_ph_mask(_mm_castsi128_ph(a.v), _mm_setzero_ph(), _CMP_LT_OQ);
  return f16x8{_mm_castph_si128(_mm_mask_blend_ph(
      negative, res, _mm_set1_ph(std::numeric_limits<float>::quiet_NaN())))};
}
#endif
#else
YNN_ALWAYS_INLINE f32x4 floor_log2(f32x4 a) {
  __m128 sign_mask = _mm_set1_ps(-0.0f);
  __m128 is_zero = _mm_cmpeq_ps(a.v, _mm_setzero_ps());
  a.v = _mm_or_ps(_mm_and_ps(is_zero, sign_mask), a.v);

  __m128i sign_and_exp_mask = _mm_set1_epi32(0xFF800000);
  __m128i exp = _mm_and_si128(_mm_castps_si128(a.v), sign_and_exp_mask);

  __m128 infinity = _mm_set1_ps(std::numeric_limits<float>::infinity());
  __m128 is_inf_or_nan =
      _mm_or_ps(_mm_cmpeq_ps(a.v, infinity), _mm_cmpunord_ps(a.v, a.v));

  exp = _mm_srai_epi32(exp, 8);

  __m128 bias_256 = _mm_set1_ps(256.0f);
  __m128 bias_383 = _mm_set1_ps(383.0f);
  __m128 res = _mm_sub_ps(_mm_or_ps(bias_256, _mm_castsi128_ps(exp)), bias_383);
  return f32x4{_mm_or_ps(_mm_andnot_ps(is_inf_or_nan, res),
                         _mm_and_ps(is_inf_or_nan, a.v))};
}
YNN_ALWAYS_INLINE f64x2 floor_log2(f64x2 a) {
  __m128d sign_mask = _mm_set1_pd(-0.0);
  __m128d is_zero = _mm_cmpeq_pd(a.v, _mm_setzero_pd());
  a.v = _mm_or_pd(_mm_and_pd(is_zero, sign_mask), a.v);

  __m128i sign_and_exp_mask = _mm_set1_epi64x(0xFFF0000000000000);
  __m128i exp = _mm_and_si128(_mm_castpd_si128(a.v), sign_and_exp_mask);

  __m128d infinity = _mm_set1_pd(std::numeric_limits<double>::infinity());
  __m128d is_inf_or_nan =
      _mm_or_pd(_mm_cmpeq_pd(a.v, infinity), _mm_cmpunord_pd(a.v, a.v));

  exp = _mm_srai_epi32(exp, 11);

  __m128d bias_2048 = _mm_set1_pd(2048.0);
  __m128d bias_3071 = _mm_set1_pd(3071.0);
  __m128d res =
      _mm_sub_pd(_mm_or_pd(bias_2048, _mm_castsi128_pd(exp)), bias_3071);
  return f64x2{_mm_or_pd(_mm_andnot_pd(is_inf_or_nan, res),
                         _mm_and_pd(is_inf_or_nan, a.v))};
}
#endif  // YNN_ARCH_X86_AVX512

YNN_ALWAYS_INLINE f32x4 exp2_round(f32x4 a) {
  const __m128 magic = _mm_set1_ps(127.0f + static_cast<float>(1 << 23));
  const __m128 res_bits = _mm_add_ps(a.v, magic);
  return f32x4{
      _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(res_bits), 23))};
}
YNN_ALWAYS_INLINE f64x2 exp2_round(f64x2 a) {
  const __m128d magic = _mm_set1_pd(1023.0 + static_cast<double>(1ll << 52));
  const __m128d res_bits = _mm_add_pd(a.v, magic);
  return f64x2{
      _mm_castsi128_pd(_mm_slli_epi64(_mm_castpd_si128(res_bits), 52))};
}

#ifdef YNN_ARCH_FP16_ARITHMETIC
YNN_ALWAYS_INLINE f16x8 exp2_round(f16x8 a) {
  const __m128h magic = _mm_set1_ph(15.0f + static_cast<float>(1 << 10));
  const __m128h res_bits = _mm_add_ph(_mm_castsi128_ph(a.v), magic);
  return f16x8{_mm_slli_epi16(_mm_castph_si128(res_bits), 10)};
}
#endif

#ifdef YNN_ARCH_X86_FMA3
YNN_ALWAYS_INLINE f64x2 fma(f64x2 a, f64x2 b, f64x2 acc) {
  return f64x2{_mm_fmadd_pd(a.v, b.v, acc.v)};
}
YNN_ALWAYS_INLINE f32x4 fma(f32x4 a, f32x4 b, f32x4 acc) {
  return f32x4{_mm_fmadd_ps(a.v, b.v, acc.v)};
}
#define YNN_HAVE_FMA
#endif

#ifdef YNN_ARCH_FP16_ARITHMETIC
YNN_ALWAYS_INLINE f16x8 fma(f16x8 a, f16x8 b, f16x8 acc) {
  return f16x8{_mm_castph_si128(_mm_fmadd_ph(
      _mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v), _mm_castsi128_ph(acc.v)))};
}
#endif

YNN_ALWAYS_INLINE s32x4 operator==(f32x4 a, f32x4 b) {
  return s32x4{_mm_castps_si128(_mm_cmpeq_ps(a.v, b.v))};
}
YNN_ALWAYS_INLINE s32x4 operator!=(f32x4 a, f32x4 b) {
  return s32x4{_mm_castps_si128(_mm_cmpneq_ps(a.v, b.v))};
}
YNN_ALWAYS_INLINE s32x4 operator<(f32x4 a, f32x4 b) {
  return s32x4{_mm_castps_si128(_mm_cmplt_ps(a.v, b.v))};
}
YNN_ALWAYS_INLINE s32x4 operator<=(f32x4 a, f32x4 b) {
  return s32x4{_mm_castps_si128(_mm_cmple_ps(a.v, b.v))};
}
YNN_ALWAYS_INLINE s32x4 operator>(f32x4 a, f32x4 b) {
  return s32x4{_mm_castps_si128(_mm_cmpgt_ps(a.v, b.v))};
}
YNN_ALWAYS_INLINE s32x4 operator>=(f32x4 a, f32x4 b) {
  return s32x4{_mm_castps_si128(_mm_cmpge_ps(a.v, b.v))};
}

YNN_ALWAYS_INLINE s64x2 operator==(f64x2 a, f64x2 b) {
  return s64x2{_mm_castpd_si128(_mm_cmpeq_pd(a.v, b.v))};
}
YNN_ALWAYS_INLINE s64x2 operator!=(f64x2 a, f64x2 b) {
  return s64x2{_mm_castpd_si128(_mm_cmpneq_pd(a.v, b.v))};
}
YNN_ALWAYS_INLINE s64x2 operator<(f64x2 a, f64x2 b) {
  return s64x2{_mm_castpd_si128(_mm_cmplt_pd(a.v, b.v))};
}
YNN_ALWAYS_INLINE s64x2 operator<=(f64x2 a, f64x2 b) {
  return s64x2{_mm_castpd_si128(_mm_cmple_pd(a.v, b.v))};
}
YNN_ALWAYS_INLINE s64x2 operator>(f64x2 a, f64x2 b) {
  return s64x2{_mm_castpd_si128(_mm_cmpgt_pd(a.v, b.v))};
}
YNN_ALWAYS_INLINE s64x2 operator>=(f64x2 a, f64x2 b) {
  return s64x2{_mm_castpd_si128(_mm_cmpge_pd(a.v, b.v))};
}

YNN_ALWAYS_INLINE s32x4 operator==(s32x4 a, s32x4 b) {
  return s32x4{_mm_cmpeq_epi32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x4 operator>(s32x4 a, s32x4 b) {
  return s32x4{_mm_cmpgt_epi32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x4 operator<(s32x4 a, s32x4 b) { return b > a; }

YNN_ALWAYS_INLINE s16x8 operator==(s16x8 a, s16x8 b) {
  return s16x8{_mm_cmpeq_epi16(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 operator>(s16x8 a, s16x8 b) {
  return s16x8{_mm_cmpgt_epi16(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 operator<(s16x8 a, s16x8 b) { return b > a; }

YNN_ALWAYS_INLINE s8x16 operator==(s8x16 a, s8x16 b) {
  return s8x16{_mm_cmpeq_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 operator>(s8x16 a, s8x16 b) {
  return s8x16{_mm_cmpgt_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 operator<(s8x16 a, s8x16 b) { return b > a; }

YNN_ALWAYS_INLINE s32x4 isnan(f32x4 a) {
  return s32x4{_mm_castps_si128(_mm_cmpunord_ps(a.v, a.v))};
}
YNN_ALWAYS_INLINE s64x2 isnan(f64x2 a) {
  return s64x2{_mm_castpd_si128(_mm_cmpunord_pd(a.v, a.v))};
}
YNN_ALWAYS_INLINE s32x4 isinf(f32x4 a) {
  __m128 mask = _mm_set1_ps(-0.0f);
  __m128 inf = _mm_set1_ps(std::numeric_limits<float>::infinity());
  return s32x4{_mm_castps_si128(_mm_cmpeq_ps(_mm_andnot_ps(mask, a.v), inf))};
}
YNN_ALWAYS_INLINE s64x2 isinf(f64x2 a) {
  __m128d mask = _mm_set1_pd(-0.0);
  __m128d inf = _mm_set1_pd(std::numeric_limits<double>::infinity());
  return s64x2{_mm_castpd_si128(_mm_cmpeq_pd(_mm_andnot_pd(mask, a.v), inf))};
}
YNN_ALWAYS_INLINE s32x4 isfinite(f32x4 a) {
  __m128 mask = _mm_set1_ps(-0.0f);
  __m128 inf = _mm_set1_ps(std::numeric_limits<float>::infinity());
  return s32x4{_mm_castps_si128(_mm_cmplt_ps(_mm_andnot_ps(mask, a.v), inf))};
}
YNN_ALWAYS_INLINE s64x2 isfinite(f64x2 a) {
  __m128d mask = _mm_set1_pd(-0.0);
  __m128d inf = _mm_set1_pd(std::numeric_limits<double>::infinity());
  return s64x2{_mm_castpd_si128(_mm_cmplt_pd(_mm_andnot_pd(mask, a.v), inf))};
}

#ifdef YNN_ARCH_FP16_ARITHMETIC
YNN_ALWAYS_INLINE __mmask8 operator==(f16x8 a, f16x8 b) {
  return _mm_cmp_ph_mask(_mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v),
                         _CMP_EQ_OQ);
}
YNN_ALWAYS_INLINE __mmask8 operator!=(f16x8 a, f16x8 b) {
  return _mm_cmp_ph_mask(_mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v),
                         _CMP_NEQ_OQ);
}
YNN_ALWAYS_INLINE __mmask8 operator<(f16x8 a, f16x8 b) {
  return _mm_cmp_ph_mask(_mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v),
                         _CMP_LT_OQ);
}
YNN_ALWAYS_INLINE __mmask8 operator<=(f16x8 a, f16x8 b) {
  return _mm_cmp_ph_mask(_mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v),
                         _CMP_LE_OQ);
}
YNN_ALWAYS_INLINE __mmask8 operator>(f16x8 a, f16x8 b) {
  return _mm_cmp_ph_mask(_mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v),
                         _CMP_GT_OQ);
}
YNN_ALWAYS_INLINE __mmask8 operator>=(f16x8 a, f16x8 b) {
  return _mm_cmp_ph_mask(_mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v),
                         _CMP_GE_OQ);
}
YNN_ALWAYS_INLINE __mmask8 isnan(f16x8 a) {
  return _mm_fpclass_ph_mask(_mm_castsi128_ph(a.v), 0x81);
}
YNN_ALWAYS_INLINE __mmask8 isinf(f16x8 a) {
  return _mm_fpclass_ph_mask(_mm_castsi128_ph(a.v), 0x18);
}
YNN_ALWAYS_INLINE __mmask8 isfinite(f16x8 a) {
  return ~_mm_fpclass_ph_mask(_mm_castsi128_ph(a.v), 0x99);
}
#endif

YNN_ALWAYS_INLINE double horizontal_sum(f64x2 a) {
  return _mm_cvtsd_f64(_mm_add_sd(a.v, _mm_shuffle_pd(a.v, a.v, 1)));
}
YNN_ALWAYS_INLINE double horizontal_max(f64x2 a) {
  return _mm_cvtsd_f64(_mm_max_sd(a.v, _mm_shuffle_pd(a.v, a.v, 1)));
}
YNN_ALWAYS_INLINE double horizontal_min(f64x2 a) {
  return _mm_cvtsd_f64(_mm_min_sd(a.v, _mm_shuffle_pd(a.v, a.v, 1)));
}

YNN_ALWAYS_INLINE float horizontal_sum(f32x4 a) {
  const __m128 sum_lanes = _mm_add_ps(a.v, _mm_movehl_ps(a.v, a.v));
  return _mm_cvtss_f32(
      _mm_add_ss(sum_lanes, _mm_shuffle_ps(sum_lanes, sum_lanes, 1)));
}
YNN_ALWAYS_INLINE float horizontal_max(f32x4 a) {
  const __m128 max_lanes = _mm_max_ps(a.v, _mm_movehl_ps(a.v, a.v));
  return _mm_cvtss_f32(
      _mm_max_ss(max_lanes, _mm_shuffle_ps(max_lanes, max_lanes, 1)));
}
YNN_ALWAYS_INLINE float horizontal_min(f32x4 a) {
  const __m128 min_lanes = _mm_min_ps(a.v, _mm_movehl_ps(a.v, a.v));
  return _mm_cvtss_f32(
      _mm_min_ss(min_lanes, _mm_shuffle_ps(min_lanes, min_lanes, 1)));
}

YNN_ALWAYS_INLINE int32_t horizontal_sum(s32x4 a) {
  const __m128i sum_lanes = _mm_add_epi32(a.v, _mm_unpackhi_epi64(a.v, a.v));
  return _mm_cvtsi128_si32(_mm_add_epi32(
      sum_lanes, _mm_shuffle_epi32(sum_lanes, _MM_SHUFFLE(0, 0, 0, 1))));
}

YNN_ALWAYS_INLINE int16_t horizontal_max(s16x8 a) {
  const __m128i max4 = _mm_max_epi16(a.v, _mm_srli_si128(a.v, 8));
  const __m128i max2 = _mm_max_epi16(max4, _mm_srli_si128(max4, 4));
  return static_cast<int16_t>(
      _mm_cvtsi128_si32(_mm_max_epi16(max2, _mm_srli_si128(max2, 2))));
}
YNN_ALWAYS_INLINE int16_t horizontal_min(s16x8 a) {
  const __m128i min4 = _mm_min_epi16(a.v, _mm_srli_si128(a.v, 8));
  const __m128i min2 = _mm_min_epi16(min4, _mm_srli_si128(min4, 4));
  return static_cast<int16_t>(
      _mm_cvtsi128_si32(_mm_min_epi16(min2, _mm_srli_si128(min2, 2))));
}

YNN_ALWAYS_INLINE uint8_t horizontal_max(u8x16 a) {
  const __m128i max8 = _mm_max_epu8(a.v, _mm_srli_si128(a.v, 8));
  const __m128i max4 = _mm_max_epu8(max8, _mm_srli_si128(max8, 4));
  const __m128i max2 = _mm_max_epu8(max4, _mm_srli_si128(max4, 2));
  const __m128i max1 = _mm_max_epu8(max2, _mm_srli_si128(max2, 1));
  return (uint8_t)_mm_cvtsi128_si32(max1);
}
YNN_ALWAYS_INLINE uint8_t horizontal_min(u8x16 a) {
  const __m128i min8 = _mm_min_epu8(a.v, _mm_srli_si128(a.v, 8));
  const __m128i min4 = _mm_min_epu8(min8, _mm_srli_si128(min8, 4));
  const __m128i min2 = _mm_min_epu8(min4, _mm_srli_si128(min4, 2));
  const __m128i min1 = _mm_min_epu8(min2, _mm_srli_si128(min2, 1));
  return (uint8_t)_mm_cvtsi128_si32(min1);
}

#ifdef YNN_ARCH_X86_SSE41
YNN_ALWAYS_INLINE int8_t horizontal_max(s8x16 a) {
  const __m128i max8 = _mm_max_epi8(a.v, _mm_srli_si128(a.v, 8));
  const __m128i max4 = _mm_max_epi8(max8, _mm_srli_si128(max8, 4));
  const __m128i max2 = _mm_max_epi8(max4, _mm_srli_si128(max4, 2));
  const __m128i max1 = _mm_max_epi8(max2, _mm_srli_si128(max2, 1));
  return static_cast<int8_t>(_mm_cvtsi128_si32(max1));
}
YNN_ALWAYS_INLINE int8_t horizontal_min(s8x16 a) {
  const __m128i min8 = _mm_min_epi8(a.v, _mm_srli_si128(a.v, 8));
  const __m128i min4 = _mm_min_epi8(min8, _mm_srli_si128(min8, 4));
  const __m128i min2 = _mm_min_epi8(min4, _mm_srli_si128(min4, 2));
  const __m128i min1 = _mm_min_epi8(min2, _mm_srli_si128(min2, 1));
  return static_cast<int8_t>(_mm_cvtsi128_si32(min1));
}
YNN_ALWAYS_INLINE int32_t horizontal_max(s32x4 a) {
  const __m128i max4 = _mm_max_epi32(a.v, _mm_srli_si128(a.v, 8));
  return _mm_cvtsi128_si32(_mm_max_epi32(max4, _mm_srli_si128(max4, 4)));
}
YNN_ALWAYS_INLINE int32_t horizontal_min(s32x4 a) {
  const __m128i min4 = _mm_min_epi32(a.v, _mm_srli_si128(a.v, 8));
  return _mm_cvtsi128_si32(_mm_min_epi32(min4, _mm_srli_si128(min4, 4)));
}
#endif  // YNN_ARCH_X86_SSE41

#ifdef YNN_ARCH_X86_SSE41
YNN_ALWAYS_INLINE f32x4 select(s32x4 cond, f32x4 a, f32x4 b) {
  return f32x4{_mm_blendv_ps(b.v, a.v, _mm_castsi128_ps(cond.v))};
}
YNN_ALWAYS_INLINE f64x2 select(s64x2 cond, f64x2 a, f64x2 b) {
  return f64x2{_mm_blendv_pd(b.v, a.v, _mm_castsi128_pd(cond.v))};
}
YNN_ALWAYS_INLINE s32x4 select(s32x4 cond, s32x4 a, s32x4 b) {
  return s32x4{_mm_blendv_epi8(b.v, a.v, cond.v)};
}
YNN_ALWAYS_INLINE u32x4 select(s32x4 cond, u32x4 a, u32x4 b) {
  return u32x4{_mm_blendv_epi8(b.v, a.v, cond.v)};
}
YNN_ALWAYS_INLINE s16x8 select(s16x8 cond, s16x8 a, s16x8 b) {
  return s16x8{_mm_blendv_epi8(b.v, a.v, cond.v)};
}
YNN_ALWAYS_INLINE u16x8 select(s16x8 cond, u16x8 a, u16x8 b) {
  return u16x8{_mm_blendv_epi8(b.v, a.v, cond.v)};
}
YNN_ALWAYS_INLINE s8x16 select(s8x16 cond, s8x16 a, s8x16 b) {
  return s8x16{_mm_blendv_epi8(b.v, a.v, cond.v)};
}
YNN_ALWAYS_INLINE u8x16 select(s8x16 cond, u8x16 a, u8x16 b) {
  return u8x16{_mm_blendv_epi8(b.v, a.v, cond.v)};
}
#else
YNN_ALWAYS_INLINE f32x4 select(s32x4 cond, f32x4 a, f32x4 b) {
  __m128 m = _mm_castsi128_ps(cond.v);
  return f32x4{_mm_or_ps(_mm_and_ps(m, a.v), _mm_andnot_ps(m, b.v))};
}
YNN_ALWAYS_INLINE f64x2 select(s64x2 cond, f64x2 a, f64x2 b) {
  __m128d m = _mm_castsi128_pd(cond.v);
  return f64x2{_mm_or_pd(_mm_and_pd(m, a.v), _mm_andnot_pd(m, b.v))};
}
YNN_ALWAYS_INLINE s32x4 select(s32x4 cond, s32x4 a, s32x4 b) {
  return s32x4{
      _mm_or_si128(_mm_and_si128(cond.v, a.v), _mm_andnot_si128(cond.v, b.v))};
}
YNN_ALWAYS_INLINE u32x4 select(s32x4 cond, u32x4 a, u32x4 b) {
  return u32x4{
      _mm_or_si128(_mm_and_si128(cond.v, a.v), _mm_andnot_si128(cond.v, b.v))};
}
YNN_ALWAYS_INLINE s16x8 select(s16x8 cond, s16x8 a, s16x8 b) {
  return s16x8{
      _mm_or_si128(_mm_and_si128(cond.v, a.v), _mm_andnot_si128(cond.v, b.v))};
}
YNN_ALWAYS_INLINE u16x8 select(s16x8 cond, u16x8 a, u16x8 b) {
  return u16x8{
      _mm_or_si128(_mm_and_si128(cond.v, a.v), _mm_andnot_si128(cond.v, b.v))};
}
YNN_ALWAYS_INLINE s8x16 select(s8x16 cond, s8x16 a, s8x16 b) {
  return s8x16{
      _mm_or_si128(_mm_and_si128(cond.v, a.v), _mm_andnot_si128(cond.v, b.v))};
}
YNN_ALWAYS_INLINE u8x16 select(s8x16 cond, u8x16 a, u8x16 b) {
  return u8x16{
      _mm_or_si128(_mm_and_si128(cond.v, a.v), _mm_andnot_si128(cond.v, b.v))};
}
#endif  // YNN_ARCH_X86_SSE41

#ifdef YNN_ARCH_FP16_ARITHMETIC
YNN_ALWAYS_INLINE f16x8 select(__mmask8 cond, f16x8 a, f16x8 b) {
  return f16x8{_mm_castph_si128(
      _mm_mask_blend_ph(cond, _mm_castsi128_ph(b.v), _mm_castsi128_ph(a.v)))};
}
#endif

YNN_ALWAYS_INLINE std::tuple<u8x16, u8x16> interleave(
    std::integral_constant<size_t, 64>, u8x16 x0, u8x16 x1) {
  return {u8x16{_mm_unpacklo_epi64(x0.v, x1.v)},
          u8x16{_mm_unpackhi_epi64(x0.v, x1.v)}};
}
YNN_ALWAYS_INLINE std::tuple<u8x16, u8x16> interleave(
    std::integral_constant<size_t, 32>, u8x16 x0, u8x16 x1) {
  return {u8x16{_mm_unpacklo_epi32(x0.v, x1.v)},
          u8x16{_mm_unpackhi_epi32(x0.v, x1.v)}};
}
YNN_ALWAYS_INLINE std::tuple<u8x16, u8x16> interleave(
    std::integral_constant<size_t, 16>, u8x16 x0, u8x16 x1) {
  return {u8x16{_mm_unpacklo_epi16(x0.v, x1.v)},
          u8x16{_mm_unpackhi_epi16(x0.v, x1.v)}};
}
YNN_ALWAYS_INLINE std::tuple<u8x16, u8x16> interleave(
    std::integral_constant<size_t, 8>, u8x16 x0, u8x16 x1) {
  return {u8x16{_mm_unpacklo_epi8(x0.v, x1.v)},
          u8x16{_mm_unpackhi_epi8(x0.v, x1.v)}};
}
YNN_ALWAYS_INLINE std::tuple<u8x16, u8x16> interleave(
    std::integral_constant<size_t, 4>, u8x16 x0, u8x16 x1) {
  __m128i even0 = _mm_and_si128(x0.v, _mm_set1_epi8(0x0f));
  __m128i even1 = _mm_and_si128(x1.v, _mm_set1_epi8(0x0f));
  __m128i odd0 = _mm_and_si128(x0.v, _mm_set1_epi8(0xf0));
  __m128i odd1 = _mm_and_si128(x1.v, _mm_set1_epi8(0xf0));
  return interleave(std::integral_constant<size_t, 8>{},
                    u8x16{_mm_or_si128(_mm_slli_epi16(even1, 4), even0)},
                    u8x16{_mm_or_si128(odd1, _mm_srli_epi16(odd0, 4))});
}
YNN_ALWAYS_INLINE std::tuple<u8x16, u8x16> interleave(
    std::integral_constant<size_t, 2>, u8x16 x0, u8x16 x1) {
  __m128i even0 = _mm_and_si128(x0.v, _mm_set1_epi8(0x33));
  __m128i even1 = _mm_and_si128(x1.v, _mm_set1_epi8(0x33));
  __m128i odd0 = _mm_and_si128(x0.v, _mm_set1_epi8(0xcc));
  __m128i odd1 = _mm_and_si128(x1.v, _mm_set1_epi8(0xcc));
  return interleave(std::integral_constant<size_t, 4>{},
                    u8x16{_mm_or_si128(_mm_slli_epi16(even1, 2), even0)},
                    u8x16{_mm_or_si128(odd1, _mm_srli_epi16(odd0, 2))});
}

YNN_ALWAYS_INLINE f32x4 cast(s32x4 x, float) {
  return f32x4{_mm_cvtepi32_ps(x.v)};
}
YNN_ALWAYS_INLINE s32x4 cast(f32x4 x, int32_t) {
  const __m128 threshold = _mm_set1_ps(2147483520.0f);
  const __m128 mask = _mm_cmpgt_ps(x.v, threshold);
  const __m128i res = _mm_cvtps_epi32(x.v);
  const __m128i imask = _mm_castps_si128(mask);
  return s32x4{_mm_or_si128(_mm_andnot_si128(imask, res),
                             _mm_and_si128(imask, _mm_set1_epi32(0x7fffffff)))};
}

#ifndef YNN_ARCH_X86_AVX

// These casts have a better implementation in AVX

using f32x8 = vec<float, 8>;
using s32x8 = vec<int32_t, 8>;
using s16x16 = vec<int16_t, 16>;
using bf16x16 = vec<bfloat16, 16>;
using f16x16 = vec<half, 16>;
using s8x32 = vec<int8_t, 32>;
using u8x32 = vec<uint8_t, 32>;
using f64x4 = vec<double, 4>;
using f32x16 = vec<float, 16>;
using s32x16 = vec<int32_t, 16>;

YNN_ALWAYS_INLINE f32x8 cast(bf16x8 b, float) {
  __m128i zero = _mm_setzero_si128();

  return {
      f32x4{_mm_castsi128_ps(_mm_unpacklo_epi16(zero, b.v))},
      f32x4{_mm_castsi128_ps(_mm_unpackhi_epi16(zero, b.v))},
  };
}

#ifdef YNN_ARCH_X86_SSE41

YNN_ALWAYS_INLINE s32x16 cast(s8x16 a, int32_t) {
  return {
      {s32x4{_mm_cvtepi8_epi32(a.v)},
       s32x4{_mm_cvtepi8_epi32(_mm_srli_si128(a.v, 4))}},
      {s32x4{_mm_cvtepi8_epi32(_mm_srli_si128(a.v, 8))},
       s32x4{_mm_cvtepi8_epi32(_mm_srli_si128(a.v, 12))}},
  };
}

YNN_ALWAYS_INLINE s32x16 cast(u8x16 a, int32_t) {
  return {
      {s32x4{_mm_cvtepu8_epi32(a.v)},
       s32x4{_mm_cvtepu8_epi32(_mm_srli_si128(a.v, 4))}},
      {s32x4{_mm_cvtepu8_epi32(_mm_srli_si128(a.v, 8))},
       s32x4{_mm_cvtepu8_epi32(_mm_srli_si128(a.v, 12))}},
  };
}

#else

YNN_ALWAYS_INLINE s32x16 cast(s8x16 a, int32_t) {
  __m128i i8_lo = _mm_unpacklo_epi8(a.v, a.v);
  __m128i i8_hi = _mm_unpackhi_epi8(a.v, a.v);

  return {
      {s32x4{_mm_srai_epi32(_mm_unpacklo_epi16(i8_lo, i8_lo), 24)},
       s32x4{_mm_srai_epi32(_mm_unpackhi_epi16(i8_lo, i8_lo), 24)}},
      {s32x4{_mm_srai_epi32(_mm_unpacklo_epi16(i8_hi, i8_hi), 24)},
       s32x4{_mm_srai_epi32(_mm_unpackhi_epi16(i8_hi, i8_hi), 24)}},
  };
}

YNN_ALWAYS_INLINE s32x16 cast(u8x16 a, int32_t) {
  const __m128i zero = _mm_setzero_si128();
  __m128i i16_lo = _mm_unpacklo_epi8(a.v, zero);
  __m128i i16_hi = _mm_unpackhi_epi8(a.v, zero);

  return {
      {s32x4{_mm_unpacklo_epi16(i16_lo, zero)},
       s32x4{_mm_unpackhi_epi16(i16_lo, zero)}},
      {s32x4{_mm_unpacklo_epi16(i16_hi, zero)},
       s32x4{_mm_unpackhi_epi16(i16_hi, zero)}},
  };
}

#endif  // YNN_ARCH_X86_SSE41

YNN_ALWAYS_INLINE f64x4 cast(f32x4 x, double) {
  return {f64x2{_mm_cvtps_pd(x.v)},
          f64x2{_mm_cvtps_pd(_mm_movehl_ps(x.v, x.v))}};
}
YNN_ALWAYS_INLINE f32x4 cast(f64x4 x, float) {
  return f32x4{_mm_movelh_ps(_mm_cvtpd_ps(x[0].v), _mm_cvtpd_ps(x[1].v))};
}

YNN_ALWAYS_INLINE s16x8 cast(s32x8 a, int16_t) {
  return s16x8{_mm_packs_epi32(lo(a).v, hi(a).v)};
}

YNN_ALWAYS_INLINE s8x16 cast(s16x16 a, int8_t) {
  return s8x16{_mm_packs_epi16(lo(a).v, hi(a).v)};
}

YNN_ALWAYS_INLINE u8x16 cast(s16x16 a, uint8_t) {
  return u8x16{_mm_packus_epi16(lo(a).v, hi(a).v)};
}

YNN_ALWAYS_INLINE s16x8 cast(f32x8 f, int16_t) {
  const s32x4 i0 = cast(lo(f), int32_t());
  const s32x4 i1 = cast(hi(f), int32_t());
  return cast(s32x8(i0, i1), int16_t());
}

YNN_ALWAYS_INLINE s8x16 cast(f32x16 f, int8_t) {
  const s16x8 i01 = cast(lo(f), int16_t());
  const s16x8 i23 = cast(hi(f), int16_t());
  return cast(s16x16(i01, i23), int8_t());
}

YNN_ALWAYS_INLINE u8x16 cast(f32x16 f, uint8_t) {
  const s32x4 i0 = cast(lo(lo(f)), int32_t());
  const s32x4 i1 = cast(hi(lo(f)), int32_t());
  const s32x4 i2 = cast(lo(hi(f)), int32_t());
  const s32x4 i3 = cast(hi(hi(f)), int32_t());
  const __m128i i01_16 = _mm_packs_epi32(i0.v, i1.v);
  const __m128i i23_16 = _mm_packs_epi32(i2.v, i3.v);
  return u8x16{_mm_packus_epi16(i01_16, i23_16)};
}

#endif  // YNN_ARCH_X86_AVX

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_VEC128_BASE_H_
