// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_SSE2_BASE_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_SSE2_BASE_H_

#include <immintrin.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/vec.h"

namespace ynn {

namespace simd {

// See vec.h for architecture independent comments.

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
struct vec<int16_t, 8> {
  using value_type = int16_t;
  static constexpr std::integral_constant<size_t, 8> N = {};

  vec() = default;
  explicit vec(__m128i v) : v(v) {}
  vec(int16_t x) : v(_mm_set1_epi16(x)) {}  // NOLINT

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

using f32x4 = vec<float, 4>;
using s32x4 = vec<int32_t, 4>;
using bf16x8 = vec<bfloat16, 8>;
using f16x8 = vec<half, 8>;
using s16x8 = vec<int16_t, 8>;
using u8x16 = vec<uint8_t, 16>;
using s8x16 = vec<int8_t, 16>;

namespace internal {

// These overloads are x86-specific helpers for implementing templated
// interleave/transpose helpers that are not x86-specific.
YNN_ALWAYS_INLINE f32x4 unpacklo(f32x4 a, f32x4 b) {
  return f32x4{_mm_unpacklo_ps(a.v, b.v)};
}
YNN_ALWAYS_INLINE f32x4 unpackhi(f32x4 a, f32x4 b) {
  return f32x4{_mm_unpackhi_ps(a.v, b.v)};
}

YNN_ALWAYS_INLINE s32x4 unpacklo(s32x4 a, s32x4 b) {
  return s32x4{_mm_unpacklo_epi32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x4 unpackhi(s32x4 a, s32x4 b) {
  return s32x4{_mm_unpackhi_epi32(a.v, b.v)};
}

YNN_ALWAYS_INLINE s8x16 unpacklo(s8x16 a, s8x16 b) {
  return s8x16{_mm_unpacklo_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 unpackhi(s8x16 a, s8x16 b) {
  return s8x16{_mm_unpackhi_epi8(a.v, b.v)};
}

YNN_ALWAYS_INLINE u8x16 unpacklo(u8x16 a, u8x16 b) {
  return u8x16{_mm_unpacklo_epi8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 unpackhi(u8x16 a, u8x16 b) {
  return u8x16{_mm_unpackhi_epi8(a.v, b.v)};
}

YNN_ALWAYS_INLINE f32x4 movehl(f32x4 a, f32x4 b) {
  return f32x4{_mm_movehl_ps(a.v, b.v)};
}
YNN_ALWAYS_INLINE f32x4 movelh(f32x4 a, f32x4 b) {
  return f32x4{_mm_movelh_ps(a.v, b.v)};
}

YNN_ALWAYS_INLINE s32x4 movehl(s32x4 a, s32x4 b) {
  return s32x4{_mm_castps_si128(
      _mm_movehl_ps(_mm_castsi128_ps(a.v), _mm_castsi128_ps(b.v)))};
}
YNN_ALWAYS_INLINE s32x4 movelh(s32x4 a, s32x4 b) {
  return s32x4{_mm_castps_si128(
      _mm_movelh_ps(_mm_castsi128_ps(a.v), _mm_castsi128_ps(b.v)))};
}

}  // namespace internal

YNN_ALWAYS_INLINE f32x4 load_aligned(const float* ptr, f32x4,
                                     decltype(f32x4::N) = {}) {
  return f32x4{_mm_load_ps(ptr)};
}
YNN_ALWAYS_INLINE s32x4 load_aligned(const int32_t* ptr, s32x4,
                                     decltype(s32x4::N) = {}) {
  return s32x4{_mm_load_si128(reinterpret_cast<const __m128i*>(ptr))};
}
YNN_ALWAYS_INLINE bf16x8 load_aligned(const bfloat16* ptr, bf16x8,
                                      decltype(bf16x8::N) = {}) {
  return bf16x8{_mm_load_si128(reinterpret_cast<const __m128i*>(ptr))};
}
YNN_ALWAYS_INLINE f16x8 load_aligned(const half* ptr, f16x8,
                                     decltype(f16x8::N) = {}) {
  return f16x8{_mm_load_si128(reinterpret_cast<const __m128i*>(ptr))};
}
YNN_ALWAYS_INLINE s16x8 load_aligned(const int16_t* ptr, s16x8,
                                     decltype(s16x8::N) = {}) {
  return s16x8{_mm_load_si128(reinterpret_cast<const __m128i*>(ptr))};
}
YNN_ALWAYS_INLINE u8x16 load_aligned(const uint8_t* ptr, u8x16,
                                     decltype(u8x16::N) = {}) {
  return u8x16{_mm_load_si128(reinterpret_cast<const __m128i*>(ptr))};
}
YNN_ALWAYS_INLINE s8x16 load_aligned(const int8_t* ptr, s8x16,
                                     decltype(s8x16::N) = {}) {
  return s8x16{_mm_load_si128(reinterpret_cast<const __m128i*>(ptr))};
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
YNN_ALWAYS_INLINE void store_aligned(int16_t* ptr, s16x8 b,
                                     decltype(s16x8::N) = {}) {
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

YNN_ALWAYS_INLINE f32x4 load(const float* ptr, f32x4, decltype(f32x4::N) = {}) {
  return f32x4{_mm_loadu_ps(ptr)};
}
YNN_ALWAYS_INLINE s32x4 load(const int32_t* ptr, s32x4,
                             decltype(s32x4::N) = {}) {
  return s32x4{_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr))};
}
YNN_ALWAYS_INLINE bf16x8 load(const bfloat16* ptr, bf16x8,
                              decltype(bf16x8::N) = {}) {
  return bf16x8{_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr))};
}
YNN_ALWAYS_INLINE f16x8 load(const half* ptr, f16x8, decltype(f16x8::N) = {}) {
  return f16x8{_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr))};
}
YNN_ALWAYS_INLINE s16x8 load(const int16_t* ptr, s16x8,
                             decltype(s16x8::N) = {}) {
  return s16x8{_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr))};
}

YNN_ALWAYS_INLINE u8x16 load(const uint8_t* ptr, u8x16,
                             decltype(u8x16::N) = {}) {
  return u8x16{_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr))};
}
YNN_ALWAYS_INLINE s8x16 load(const int8_t* ptr, s8x16,
                             decltype(s8x16::N) = {}) {
  return s8x16{_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr))};
}

YNN_ALWAYS_INLINE void store(float* ptr, f32x4 b, decltype(f32x4::N) = {}) {
  _mm_storeu_ps(ptr, b.v);
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
YNN_ALWAYS_INLINE void store(int16_t* ptr, s16x8 b, decltype(s16x8::N) = {}) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x16 b, decltype(u8x16::N) = {}) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(int8_t* ptr, s8x16 b, decltype(s8x16::N) = {}) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), b.v);
}

// Partial load/store with a non-constant number of elements.

namespace internal {

template <typename T>
vec<T, 4> partial_load_switch_x4(const T* ptr, vec<T, 4> src, size_t n) {
  assert(n < 4);
  alignas(sizeof(vec<T, 4>)) T lanes[4];
  store_aligned(lanes, src);
  switch (n) {
    case 3:
      lanes[2] = ptr[2];
      [[fallthrough]];
    case 2:
      lanes[1] = ptr[1];
      [[fallthrough]];
    case 1:
      lanes[0] = ptr[0];
      break;
    default:
      break;
  }
  return load_aligned(lanes, vec<T, 4>{});
}

YNN_ALWAYS_INLINE void storel(int32_t* ptr, __m128i v) {
  _mm_storel_pi(reinterpret_cast<__m64*>(ptr), _mm_castsi128_ps(v));
}
YNN_ALWAYS_INLINE void storel(float* ptr, __m128 v) {
  _mm_storel_pi(reinterpret_cast<__m64*>(ptr), v);
}

YNN_ALWAYS_INLINE void store_scalar(int32_t* ptr, __m128i v) {
  _mm_store_ss(reinterpret_cast<float*>(ptr), _mm_castsi128_ps(v));
}
YNN_ALWAYS_INLINE void store_scalar(float* ptr, __m128 v) {
  _mm_store_ss(ptr, v);
}

template <typename T>
void partial_store_x32x4(T* ptr, vec<T, 4> b, size_t n) {
  assert(n < 4);
  if (n & 2) {
    storel(ptr, b.v);
    ptr += 2;
    b = movehl(b, b);
  }
  if (n & 1) {
    store_scalar(ptr, b.v);
  }
}

}  // namespace internal

YNN_ALWAYS_INLINE f32x4 load(const float* ptr, f32x4 src, size_t n) {
  return internal::partial_load_switch_x4(ptr, src, n);
}
YNN_ALWAYS_INLINE s32x4 load(const int32_t* ptr, s32x4 src, size_t n) {
  return internal::partial_load_switch_x4(ptr, src, n);
}
YNN_ALWAYS_INLINE bf16x8 load(const bfloat16* ptr, bf16x8 src, size_t n) {
  return internal::partial_load_memcpy(ptr, src, n);
}
YNN_ALWAYS_INLINE f16x8 load(const half* ptr, f16x8 src, size_t n) {
  return internal::partial_load_memcpy(ptr, src, n);
}
YNN_ALWAYS_INLINE s16x8 load(const int16_t* ptr, s16x8 src, size_t n) {
  return internal::partial_load_memcpy(ptr, src, n);
}
YNN_ALWAYS_INLINE void store(float* ptr, f32x4 b, size_t n) {
  internal::partial_store_x32x4(ptr, b, n);
}
YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x4 b, size_t n) {
  internal::partial_store_x32x4(ptr, b, n);
}
YNN_ALWAYS_INLINE void store(bfloat16* ptr, bf16x8 b, size_t n) {
  internal::partial_store_memcpy(ptr, b, n);
}
YNN_ALWAYS_INLINE void store(half* ptr, f16x8 b, size_t n) {
  internal::partial_store_memcpy(ptr, b, n);
}
YNN_ALWAYS_INLINE void store(int16_t* ptr, s16x8 b, size_t n) {
  internal::partial_store_memcpy(ptr, b, n);
}

YNN_ALWAYS_INLINE u8x16 load(const uint8_t* ptr, u8x16 src, size_t n) {
  return internal::partial_load_memcpy(ptr, src, n);
}
YNN_ALWAYS_INLINE s8x16 load(const int8_t* ptr, s8x16 src, size_t n) {
  return internal::partial_load_memcpy(ptr, src, n);
}

YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x16 val, size_t n) {
  internal::partial_store_memcpy(ptr, val, n);
}
YNN_ALWAYS_INLINE void store(int8_t* ptr, s8x16 val, size_t n) {
  internal::partial_store_memcpy(ptr, val, n);
}

YNN_ALWAYS_INLINE f32x4& operator+=(f32x4& a, f32x4 b) {
  a.v = _mm_add_ps(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s32x4& operator+=(s32x4& a, s32x4 b) {
  a.v = _mm_add_epi32(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s16x8& operator+=(s16x8& a, s16x8 b) {
  a.v = _mm_add_epi16(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s8x16& operator+=(s8x16& a, s8x16 b) {
  a.v = _mm_add_epi8(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE u8x16& operator+=(u8x16& a, u8x16 b) {
  a.v = _mm_add_epi8(a.v, b.v);
  return a;
}

YNN_ALWAYS_INLINE f32x4& operator-=(f32x4& a, f32x4 b) {
  a.v = _mm_sub_ps(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s32x4& operator-=(s32x4& a, s32x4 b) {
  a.v = _mm_sub_epi32(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s16x8& operator-=(s16x8& a, s16x8 b) {
  a.v = _mm_sub_epi16(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s8x16& operator-=(s8x16& a, s8x16 b) {
  a.v = _mm_sub_epi8(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE u8x16& operator-=(u8x16& a, u8x16 b) {
  a.v = _mm_sub_epi8(a.v, b.v);
  return a;
}

YNN_ALWAYS_INLINE f32x4& operator*=(f32x4& a, f32x4 b) {
  a.v = _mm_mul_ps(a.v, b.v);
  return a;
}

YNN_ALWAYS_INLINE f32x4 operator+(f32x4 a, f32x4 b) { return a += b; }
YNN_ALWAYS_INLINE s32x4 operator+(s32x4 a, s32x4 b) { return a += b; }
YNN_ALWAYS_INLINE s16x8 operator+(s16x8 a, s16x8 b) { return a += b; }
YNN_ALWAYS_INLINE s8x16 operator+(s8x16 a, s8x16 b) { return a += b; }
YNN_ALWAYS_INLINE u8x16 operator+(u8x16 a, u8x16 b) { return a += b; }

YNN_ALWAYS_INLINE f32x4 operator-(f32x4 a, f32x4 b) { return a -= b; }
YNN_ALWAYS_INLINE s32x4 operator-(s32x4 a, s32x4 b) { return a -= b; }
YNN_ALWAYS_INLINE s16x8 operator-(s16x8 a, s16x8 b) { return a -= b; }
YNN_ALWAYS_INLINE s8x16 operator-(s8x16 a, s8x16 b) { return a -= b; }
YNN_ALWAYS_INLINE u8x16 operator-(u8x16 a, u8x16 b) { return a -= b; }

YNN_ALWAYS_INLINE f32x4 operator*(f32x4 a, f32x4 b) { return a *= b; }

YNN_ALWAYS_INLINE s16x8 operator&(s16x8 a, s16x8 b) {
  return s16x8{_mm_and_si128(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 operator>>(s16x8 a, int b) {
  return s16x8{_mm_srai_epi16(a.v, b)};
}
YNN_ALWAYS_INLINE s16x8 operator^(s16x8 a, s16x8 b) {
  return s16x8{_mm_xor_si128(a.v, b.v)};
}

YNN_ALWAYS_INLINE f32x4 min(f32x4 a, f32x4 b) {
  return f32x4{_mm_min_ps(a.v, b.v)};
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
YNN_ALWAYS_INLINE s16x8 max(s16x8 a, s16x8 b) {
  return s16x8{_mm_max_epi16(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 max(u8x16 a, u8x16 b) {
  return u8x16{_mm_max_epu8(a.v, b.v)};
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

template <typename T>
YNN_ALWAYS_INLINE std::array<vec<T, 4>, 4> transpose(
    std::array<vec<T, 4>, 4> x) {
  vec<T, 4> t0 = internal::unpacklo(x[0], x[1]);
  vec<T, 4> t1 = internal::unpacklo(x[2], x[3]);
  vec<T, 4> t2 = internal::unpackhi(x[0], x[1]);
  vec<T, 4> t3 = internal::unpackhi(x[2], x[3]);
  return {{
      {internal::movelh(t0, t1)},
      {internal::movehl(t1, t0)},
      {internal::movelh(t2, t3)},
      {internal::movehl(t3, t2)},
  }};
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_SSE2_BASE_H_
