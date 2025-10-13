// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_AVX_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_AVX_H_

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/simd/x86_sse.h"

namespace ynn {

namespace simd {

// See vec.h for architecture independent comments.

template <>
struct vec<float, 8> {
  using value_type = float;
  static constexpr std::integral_constant<size_t, 8> N = {};

  vec() = default;
  explicit vec(__m256 v) : v(v) {}
  vec(float x) : v(_mm256_set1_ps(x)) {}  // NOLINT

  __m256 v;
};

template <>
struct vec<int32_t, 8> {
  using value_type = int32_t;
  static constexpr std::integral_constant<size_t, 8> N = {};

  vec() = default;
  explicit vec(__m256i v) : v(v) {}
  vec(int32_t x) : v(_mm256_set1_epi32(x)) {}  // NOLINT

  __m256i v;
};

template <>
struct vec<bfloat16, 16> {
  using value_type = bfloat16;
  static constexpr std::integral_constant<size_t, 16> N = {};

  vec() = default;
  explicit vec(__m256i v) : v(v) {}
  vec(bfloat16 x) : v(_mm256_set1_epi16(x.to_bits())) {}  // NOLINT

  __m256i v;
};

template <>
struct vec<half, 16> {
  using value_type = half;
  static constexpr std::integral_constant<size_t, 16> N = {};

  vec() = default;
  explicit vec(__m256i v) : v(v) {}
  vec(half x) : v(_mm256_set1_epi16(x.to_bits())) {}  // NOLINT

  __m256i v;
};

template <>
struct vec<int16_t, 16> {
  using value_type = int16_t;
  static constexpr std::integral_constant<size_t, 16> N = {};

  vec() = default;
  explicit vec(__m256i v) : v(v) {}
  vec(int16_t x) : v(_mm256_set1_epi16(x)) {}  // NOLINT

  __m256i v;
};

template <>
struct vec<uint8_t, 32> {
  using value_type = uint8_t;
  static constexpr std::integral_constant<size_t, 32> N = {};

  vec() = default;
  explicit vec(__m256i v) : v(v) {}
  vec(uint8_t x) : v(_mm256_set1_epi8(x)) {}  // NOLINT

  __m256i v;
};

template <>
struct vec<int8_t, 32> {
  using value_type = int8_t;
  static constexpr std::integral_constant<size_t, 32> N = {};

  vec() = default;
  explicit vec(__m256i v) : v(v) {}
  vec(int8_t x) : v(_mm256_set1_epi8(x)) {}  // NOLINT

  __m256i v;
};

using f32x8 = vec<float, 8>;
using s32x8 = vec<int32_t, 8>;
using bf16x16 = vec<bfloat16, 16>;
using f16x16 = vec<half, 16>;
using s16x16 = vec<int16_t, 16>;
using u8x32 = vec<uint8_t, 32>;
using s8x32 = vec<int8_t, 32>;

namespace internal {

// These overloads are x86-specific helpers for implementing templated
// interleave/transpose helpers that are not x86-specific.
YNN_ALWAYS_INLINE f32x8 unpacklo(f32x8 a, f32x8 b) {
  return f32x8{_mm256_unpacklo_ps(a.v, b.v)};
}
YNN_ALWAYS_INLINE f32x8 unpackhi(f32x8 a, f32x8 b) {
  return f32x8{_mm256_unpackhi_ps(a.v, b.v)};
}

YNN_ALWAYS_INLINE s32x8 unpacklo(s32x8 a, s32x8 b) {
  return s32x8{_mm256_unpacklo_epi32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x8 unpackhi(s32x8 a, s32x8 b) {
  return s32x8{_mm256_unpackhi_epi32(a.v, b.v)};
}

YNN_ALWAYS_INLINE s8x32 unpacklo(s8x32 a, s8x32 b) {
  return s8x32{_mm256_unpacklo_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x32 unpackhi(s8x32 a, s8x32 b) {
  return s8x32{_mm256_unpackhi_epi8(a.v, b.v)};
}

YNN_ALWAYS_INLINE u8x32 unpacklo(u8x32 a, u8x32 b) {
  return u8x32{_mm256_unpacklo_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x32 unpackhi(u8x32 a, u8x32 b) {
  return u8x32{_mm256_unpackhi_epi8(a.v, b.v)};
}

}  // namespace internal

YNN_ALWAYS_INLINE f32x8 load_aligned(const float* ptr, f32x8,
                                     decltype(f32x8::N) = {}) {
  return f32x8{_mm256_load_ps(ptr)};
}
YNN_ALWAYS_INLINE s32x8 load_aligned(const int32_t* ptr, s32x8,
                                     decltype(s32x8::N) = {}) {
  return s32x8{_mm256_load_si256(reinterpret_cast<const __m256i*>(ptr))};
}
YNN_ALWAYS_INLINE bf16x16 load_aligned(const bfloat16* ptr, bf16x16,
                                       decltype(bf16x16::N) = {}) {
  return bf16x16{_mm256_load_si256(reinterpret_cast<const __m256i*>(ptr))};
}
YNN_ALWAYS_INLINE f16x16 load_aligned(const half* ptr, f16x16,
                                      decltype(f16x16::N) = {}) {
  return f16x16{_mm256_load_si256(reinterpret_cast<const __m256i*>(ptr))};
}
YNN_ALWAYS_INLINE s16x16 load_aligned(const int16_t* ptr, s16x16,
                                      decltype(s16x16::N) = {}) {
  return s16x16{_mm256_load_si256(reinterpret_cast<const __m256i*>(ptr))};
}
YNN_ALWAYS_INLINE u8x32 load_aligned(const uint8_t* ptr, u8x32,
                                     decltype(u8x32::N) = {}) {
  return u8x32{_mm256_load_si256(reinterpret_cast<const __m256i*>(ptr))};
}
YNN_ALWAYS_INLINE s8x32 load_aligned(const int8_t* ptr, s8x32,
                                     decltype(s8x32::N) = {}) {
  return s8x32{_mm256_load_si256(reinterpret_cast<const __m256i*>(ptr))};
}

YNN_ALWAYS_INLINE void store_aligned(float* ptr, f32x8 b,
                                     decltype(f32x8::N) = {}) {
  _mm256_store_ps(ptr, b.v);
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

YNN_ALWAYS_INLINE f32x8 load(const float* ptr, f32x8, decltype(f32x8::N) = {}) {
  return f32x8{_mm256_loadu_ps(ptr)};
}
YNN_ALWAYS_INLINE s32x8 load(const int32_t* ptr, s32x8,
                             decltype(s32x8::N) = {}) {
  return s32x8{_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr))};
}
YNN_ALWAYS_INLINE bf16x16 load(const bfloat16* ptr, bf16x16,
                               decltype(bf16x16::N) = {}) {
  return bf16x16{_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr))};
}
YNN_ALWAYS_INLINE f16x16 load(const half* ptr, f16x16,
                              decltype(f16x16::N) = {}) {
  return f16x16{_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr))};
}
YNN_ALWAYS_INLINE s16x16 load(const int16_t* ptr, s16x16,
                              decltype(s16x16::N) = {}) {
  return s16x16{_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr))};
}
YNN_ALWAYS_INLINE u8x32 load(const uint8_t* ptr, u8x32,
                             decltype(u8x32::N) = {}) {
  return u8x32{_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr))};
}
YNN_ALWAYS_INLINE s8x32 load(const int8_t* ptr, s8x32,
                             decltype(s8x32::N) = {}) {
  return s8x32{_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr))};
}

YNN_ALWAYS_INLINE void store(float* ptr, f32x8 b, decltype(f32x8::N) = {}) {
  _mm256_storeu_ps(ptr, b.v);
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
YNN_ALWAYS_INLINE void store(int16_t* ptr, s16x16 b, decltype(s16x16::N) = {}) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x32 b, decltype(u8x32::N) = {}) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(int8_t* ptr, s8x32 b, decltype(s8x32::N) = {}) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), b.v);
}

namespace internal {

static constexpr int32_t mask_table[16] = {-1, -1, -1, -1, -1, -1, -1, -1,
                                           0,  0,  0,  0,  0,  0,  0,  0};

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

YNN_ALWAYS_INLINE void maskstore(float* ptr, f32x8 val, __m256i mask) {
  _mm256_maskstore_ps(ptr, mask, val.v);
}
YNN_ALWAYS_INLINE void maskstore(int32_t* ptr, s32x8 val, __m256i mask) {
  _mm256_maskstore_ps(reinterpret_cast<float*>(ptr), mask,
                      _mm256_castsi256_ps(val.v));
}

// Partial load/store with a non-constant number of elements.
template <typename T>
inline vec<T, 8> partial_load_mask_x32x8(const T* ptr, vec<T, 8> src,
                                         size_t n) {
  assert(n <= 8);
  auto mask =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&mask_table[8 - n]));
  return vec<T, 8>{maskload(ptr, src.v, mask)};
}

template <typename T>
inline void partial_store_x32x8(T* ptr, vec<T, 8> val, size_t n) {
  assert(n <= 8);
  auto mask =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&mask_table[8 - n]));
  maskstore(ptr, val, mask);
}

}  // namespace internal

YNN_ALWAYS_INLINE f32x8 load(const float* ptr, f32x8 src, size_t n) {
  return internal::partial_load_mask_x32x8(ptr, src, n);
}
YNN_ALWAYS_INLINE s32x8 load(const int32_t* ptr, s32x8 src, size_t n) {
  return internal::partial_load_mask_x32x8(ptr, src, n);
}
YNN_ALWAYS_INLINE bf16x16 load(const bfloat16* ptr, bf16x16 src, size_t n) {
  return internal::partial_load_memcpy(ptr, src, n);
}
YNN_ALWAYS_INLINE f16x16 load(const half* ptr, f16x16 src, size_t n) {
  return internal::partial_load_memcpy(ptr, src, n);
}
YNN_ALWAYS_INLINE s16x16 load(const int16_t* ptr, s16x16 src, size_t n) {
  return internal::partial_load_memcpy(ptr, src, n);
}
YNN_ALWAYS_INLINE u8x32 load(const uint8_t* ptr, u8x32 src, size_t n) {
  return internal::partial_load_memcpy(ptr, src, n);
}
YNN_ALWAYS_INLINE s8x32 load(const int8_t* ptr, s8x32 src, size_t n) {
  return internal::partial_load_memcpy(ptr, src, n);
}

YNN_ALWAYS_INLINE void store(float* ptr, f32x8 val, size_t n) {
  internal::partial_store_x32x8(ptr, val, n);
}
YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x8 val, size_t n) {
  internal::partial_store_x32x8(ptr, val, n);
}
YNN_ALWAYS_INLINE void store(bfloat16* ptr, bf16x16 val, size_t n) {
  internal::partial_store_memcpy(ptr, val, n);
}
YNN_ALWAYS_INLINE void store(half* ptr, f16x16 val, size_t n) {
  internal::partial_store_memcpy(ptr, val, n);
}
YNN_ALWAYS_INLINE void store(int16_t* ptr, s16x16 val, size_t n) {
  internal::partial_store_memcpy(ptr, val, n);
}
YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x32 val, size_t n) {
  internal::partial_store_memcpy(ptr, val, n);
}
YNN_ALWAYS_INLINE void store(int8_t* ptr, s8x32 val, size_t n) {
  internal::partial_store_memcpy(ptr, val, n);
}

YNN_ALWAYS_INLINE f32x8& operator+=(f32x8& a, f32x8 b) {
  a.v = _mm256_add_ps(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s32x8& operator+=(s32x8& a, s32x8 b) {
  a.v = _mm256_add_epi32(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s8x32& operator+=(s8x32& a, s8x32 b) {
  a.v = _mm256_add_epi8(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE u8x32& operator+=(u8x32& a, u8x32 b) {
  a.v = _mm256_add_epi8(a.v, b.v);
  return a;
}

YNN_ALWAYS_INLINE f32x8& operator-=(f32x8& a, f32x8 b) {
  a.v = _mm256_sub_ps(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s32x8& operator-=(s32x8& a, s32x8 b) {
  a.v = _mm256_sub_epi32(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s8x32& operator-=(s8x32& a, s8x32 b) {
  a.v = _mm256_sub_epi8(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE u8x32& operator-=(u8x32& a, u8x32 b) {
  a.v = _mm256_sub_epi8(a.v, b.v);
  return a;
}

YNN_ALWAYS_INLINE f32x8& operator*=(f32x8& a, f32x8 b) {
  a.v = _mm256_mul_ps(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s32x8& operator*=(s32x8& a, s32x8 b) {
  a.v = _mm256_mul_epi32(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE f32x8 operator+(f32x8 a, f32x8 b) { return a += b; }
YNN_ALWAYS_INLINE s32x8 operator+(s32x8 a, s32x8 b) { return a += b; }
YNN_ALWAYS_INLINE s8x32 operator+(s8x32 a, s8x32 b) { return a += b; }
YNN_ALWAYS_INLINE u8x32 operator+(u8x32 a, u8x32 b) { return a += b; }

YNN_ALWAYS_INLINE f32x8 operator-(f32x8 a, f32x8 b) { return a -= b; }
YNN_ALWAYS_INLINE s32x8 operator-(s32x8 a, s32x8 b) { return a -= b; }
YNN_ALWAYS_INLINE s8x32 operator-(s8x32 a, s8x32 b) { return a -= b; }
YNN_ALWAYS_INLINE u8x32 operator-(u8x32 a, u8x32 b) { return a -= b; }

YNN_ALWAYS_INLINE f32x8 operator*(f32x8 a, f32x8 b) { return a *= b; }
YNN_ALWAYS_INLINE s32x8 operator*(s32x8 a, s32x8 b) { return a *= b; }

YNN_ALWAYS_INLINE s16x16 operator&(s16x16 a, int b) {
  return s16x16{_mm256_and_si256(a.v, _mm256_set1_epi16(b))};
}
YNN_ALWAYS_INLINE s16x16 operator>>(s16x16 a, int b) {
  return s16x16{_mm256_srai_epi16(a.v, b)};
}
YNN_ALWAYS_INLINE s16x16 operator^(s16x16 a, s16x16 b) {
  return s16x16{_mm256_xor_si256(a.v, b.v)};
}

YNN_ALWAYS_INLINE f32x8 min(f32x8 a, f32x8 b) {
  return f32x8{_mm256_min_ps(a.v, b.v)};
}
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

YNN_ALWAYS_INLINE f32x8 max(f32x8 a, f32x8 b) {
  return f32x8{_mm256_max_ps(a.v, b.v)};
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

YNN_ALWAYS_INLINE float horizontal_max(f32x8 a) {
  __m128 max_128 =
      _mm_max_ps(_mm256_castps256_ps128(a.v), _mm256_extractf128_ps(a.v, 1));
  __m128 max_64 = _mm_max_ps(max_128, _mm_movehl_ps(max_128, max_128));
  return _mm_cvtss_f32(_mm_max_ss(max_64, _mm_shuffle_ps(max_64, max_64, 1)));
}
YNN_ALWAYS_INLINE float horizontal_min(f32x8 a) {
  __m128 min_128 =
      _mm_min_ps(_mm256_castps256_ps128(a.v), _mm256_extractf128_ps(a.v, 1));
  __m128 min_64 = _mm_min_ps(min_128, _mm_movehl_ps(min_128, min_128));
  return _mm_cvtss_f32(_mm_min_ss(min_64, _mm_shuffle_ps(min_64, min_64, 1)));
}

YNN_ALWAYS_INLINE int16_t horizontal_max(s16x16 a) {
  const __m128i max8_vals = _mm_max_epi16(_mm256_castsi256_si128(a.v),
                                          _mm256_extracti128_si256(a.v, 1));
  const __m128i max4_vals =
      _mm_max_epi16(max8_vals, _mm_srli_si128(max8_vals, 8));
  const __m128i max2_vals =
      _mm_max_epi16(max4_vals, _mm_srli_si128(max4_vals, 4));
  return static_cast<int16_t>(_mm_cvtsi128_si32(
      _mm_max_epi16(max2_vals, _mm_srli_si128(max2_vals, 2))));
}
YNN_ALWAYS_INLINE int16_t horizontal_min(s16x16 a) {
  const __m128i min8_vals = _mm_min_epi16(_mm256_castsi256_si128(a.v),
                                          _mm256_extracti128_si256(a.v, 1));
  const __m128i min4_vals =
      _mm_min_epi16(min8_vals, _mm_srli_si128(min8_vals, 8));
  const __m128i min2_vals =
      _mm_min_epi16(min4_vals, _mm_srli_si128(min4_vals, 4));
  return static_cast<int16_t>(_mm_cvtsi128_si32(
      _mm_min_epi16(min2_vals, _mm_srli_si128(min2_vals, 2))));
}

YNN_ALWAYS_INLINE int32_t horizontal_max(s32x8 a) {
  const __m128i max8_vals = _mm_max_epi32(_mm256_castsi256_si128(a.v),
                                          _mm256_extracti128_si256(a.v, 1));
  const __m128i max4_vals =
      _mm_max_epi32(max8_vals, _mm_srli_si128(max8_vals, 8));
  return _mm_cvtsi128_si32(
      _mm_max_epi32(max4_vals, _mm_srli_si128(max4_vals, 4)));
}
YNN_ALWAYS_INLINE int32_t horizontal_min(s32x8 a) {
  const __m128i min8_vals = _mm_min_epi32(_mm256_castsi256_si128(a.v),
                                          _mm256_extracti128_si256(a.v, 1));
  const __m128i min4_vals =
      _mm_min_epi32(min8_vals, _mm_srli_si128(min8_vals, 8));
  return static_cast<int32_t>(_mm_cvtsi128_si32(
      _mm_min_epi32(min4_vals, _mm_srli_si128(min4_vals, 4))));
}

YNN_ALWAYS_INLINE int8_t horizontal_max(s8x32 a) {
  const __m128i max16 = _mm_max_epi8(_mm256_castsi256_si128(a.v),
                                     _mm256_extracti128_si256(a.v, 1));
  const __m128i max8 = _mm_max_epi8(max16, _mm_srli_si128(max16, 8));
  const __m128i max4 = _mm_max_epi8(max8, _mm_srli_si128(max8, 4));
  const __m128i max2 = _mm_max_epi8(max4, _mm_srli_si128(max4, 2));
  return static_cast<int8_t>(
      _mm_cvtsi128_si32(_mm_max_epi8(max2, _mm_srli_si128(max2, 1))));
}
YNN_ALWAYS_INLINE int8_t horizontal_min(s8x32 a) {
  const __m128i min16 = _mm_min_epi8(_mm256_castsi256_si128(a.v),
                                     _mm256_extracti128_si256(a.v, 1));
  const __m128i min8 = _mm_min_epi8(min16, _mm_srli_si128(min16, 8));
  const __m128i min4 = _mm_min_epi8(min8, _mm_srli_si128(min8, 4));
  const __m128i min2 = _mm_min_epi8(min4, _mm_srli_si128(min4, 2));
  return static_cast<int8_t>(
      _mm_cvtsi128_si32(_mm_min_epi8(min2, _mm_srli_si128(min2, 1))));
}

YNN_ALWAYS_INLINE uint8_t horizontal_max(u8x32 a) {
  const __m128i max16 = _mm_max_epu8(_mm256_castsi256_si128(a.v),
                                     _mm256_extracti128_si256(a.v, 1));
  const __m128i max8 = _mm_max_epu8(max16, _mm_srli_si128(max16, 8));
  const __m128i max4 = _mm_max_epu8(max8, _mm_srli_si128(max8, 4));
  const __m128i max2 = _mm_max_epu8(max4, _mm_srli_si128(max4, 2));
  return static_cast<uint8_t>(
      _mm_cvtsi128_si32(_mm_max_epu8(max2, _mm_srli_si128(max2, 1))));
}
YNN_ALWAYS_INLINE uint8_t horizontal_min(u8x32 a) {
  const __m128i min16 = _mm_min_epu8(_mm256_castsi256_si128(a.v),
                                     _mm256_extracti128_si256(a.v, 1));
  const __m128i min8 = _mm_min_epu8(min16, _mm_srli_si128(min16, 8));
  const __m128i min4 = _mm_min_epu8(min8, _mm_srli_si128(min8, 4));
  const __m128i min2 = _mm_min_epu8(min4, _mm_srli_si128(min4, 2));
  return static_cast<uint8_t>(
      _mm_cvtsi128_si32(_mm_min_epu8(min2, _mm_srli_si128(min2, 1))));
}

// Extract the `Index`th instance of the second type from `x`.
template <int Index>
YNN_ALWAYS_INLINE s32x4 extract(s32x8 x, s32x4) {
  return s32x4{_mm256_extracti128_si256(x.v, Index)};
}
template <int Index>
YNN_ALWAYS_INLINE f32x4 extract(f32x8 x, f32x4) {
  return f32x4{_mm256_extractf128_ps(x.v, Index)};
}
template <int Index>
YNN_ALWAYS_INLINE f16x8 extract(f16x16 x, f16x8) {
  return f16x8{
      _mm_castps_si128(_mm256_extractf128_ps(_mm256_castsi256_ps(x.v), Index))};
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_AVX_H_
