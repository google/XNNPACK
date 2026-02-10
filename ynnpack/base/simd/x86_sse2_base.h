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
YNN_ALWAYS_INLINE __m128 unpacklo_x32x4(__m128 a, __m128 b) {
  return _mm_unpacklo_ps(a, b);
}
YNN_ALWAYS_INLINE __m128 unpackhi_x32x4(__m128 a, __m128 b) {
  return _mm_unpackhi_ps(a, b);
}

YNN_ALWAYS_INLINE __m128i unpacklo_x32x4(__m128i a, __m128i b) {
  return _mm_unpacklo_epi32(a, b);
}
YNN_ALWAYS_INLINE __m128i unpackhi_x32x4(__m128i a, __m128i b) {
  return _mm_unpackhi_epi32(a, b);
}

YNN_ALWAYS_INLINE __m128i unpacklo_x8x16(__m128i a, __m128i b) {
  return _mm_unpacklo_epi8(a, b);
}
YNN_ALWAYS_INLINE __m128i unpackhi_x8x16(__m128i a, __m128i b) {
  return _mm_unpackhi_epi8(a, b);
}

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

template <typename T, size_t N>
void store_aligned(T* dst, zeros<N>) {
  memset(dst, 0, N * sizeof(T));
}

template <typename T, size_t N>
void store_aligned(T* dst, undef<N>) {}

template <typename T, typename Init>
vec<T, Init::N> partial_load_sse(const T* ptr, size_t n, Init src) {
  assert(n < 4);
  alignas(sizeof(vec<T, Init::N>)) T lanes[Init::N];
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
  return load_aligned(lanes, Init::N);
}

YNN_ALWAYS_INLINE void store_64(void* ptr, __m128i v) {
  _mm_storel_pi(reinterpret_cast<__m64*>(ptr), _mm_castsi128_ps(v));
}
YNN_ALWAYS_INLINE void store_64(void* ptr, __m128 v) {
  _mm_storel_pi(reinterpret_cast<__m64*>(ptr), v);
}

YNN_ALWAYS_INLINE void store_32(void* ptr, __m128i v) {
  _mm_store_ss(reinterpret_cast<float*>(ptr), _mm_castsi128_ps(v));
}
YNN_ALWAYS_INLINE void store_32(void* ptr, __m128 v) {
  _mm_store_ss(reinterpret_cast<float*>(ptr), v);
}

template <typename T>
static void partial_store_sse(T* ptr, vec<T, 4> b, size_t n) {
  assert(n < 4);
  if (n & 2) {
    store_64(ptr, b.v);
    ptr += 2;
    b.v = movehl(b.v, b.v);
  }
  if (n & 1) {
    store_32(ptr, b.v);
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
  return internal::partial_load_memcpy(ptr, n, src);
}
YNN_ALWAYS_INLINE f16x8 load(const half* ptr, size_t n, f16x8 src) {
  return internal::partial_load_memcpy(ptr, n, src);
}
YNN_ALWAYS_INLINE s16x8 load(const int16_t* ptr, size_t n, s16x8 src) {
  return internal::partial_load_memcpy(ptr, n, src);
}
YNN_ALWAYS_INLINE u8x16 load(const uint8_t* ptr, size_t n, u8x16 src) {
  return internal::partial_load_memcpy(ptr, n, src);
}
YNN_ALWAYS_INLINE s8x16 load(const int8_t* ptr, size_t n, s8x16 src) {
  return internal::partial_load_memcpy(ptr, n, src);
}

YNN_ALWAYS_INLINE f32x4 load(const float* ptr, size_t n, zeros<4> src) {
  return internal::partial_load_sse(ptr, n, src);
}
YNN_ALWAYS_INLINE s32x4 load(const int32_t* ptr, size_t n, zeros<4> src) {
  return internal::partial_load_sse(ptr, n, src);
}
YNN_ALWAYS_INLINE bf16x8 load(const bfloat16* ptr, size_t n, zeros<8> src) {
  return internal::partial_load_memcpy(ptr, n, bf16x8{0});
}
YNN_ALWAYS_INLINE f16x8 load(const half* ptr, size_t n, zeros<8> src) {
  return internal::partial_load_memcpy(ptr, n, f16x8{0});
}
YNN_ALWAYS_INLINE s16x8 load(const int16_t* ptr, size_t n, zeros<8> src) {
  return internal::partial_load_memcpy(ptr, n, s16x8{0});
}
YNN_ALWAYS_INLINE u8x16 load(const uint8_t* ptr, size_t n, zeros<16> src) {
  return internal::partial_load_memcpy(ptr, n, u8x16{0});
}
YNN_ALWAYS_INLINE s8x16 load(const int8_t* ptr, size_t n, zeros<16> src) {
  return internal::partial_load_memcpy(ptr, n, s8x16{0});
}

YNN_ALWAYS_INLINE f32x4 load(const float* ptr, size_t n, undef<4> src) {
  return internal::partial_load_sse(ptr, n, src);
}
YNN_ALWAYS_INLINE s32x4 load(const int32_t* ptr, size_t n, undef<4> src) {
  return internal::partial_load_sse(ptr, n, src);
}
YNN_ALWAYS_INLINE bf16x8 load(const bfloat16* ptr, size_t n, undef<8> src) {
  return internal::partial_load_memcpy(ptr, n, bf16x8{});
}
YNN_ALWAYS_INLINE f16x8 load(const half* ptr, size_t n, undef<8> src) {
  return internal::partial_load_memcpy(ptr, n, f16x8{});
}
YNN_ALWAYS_INLINE s16x8 load(const int16_t* ptr, size_t n, undef<8> src) {
  return internal::partial_load_memcpy(ptr, n, s16x8{});
}
YNN_ALWAYS_INLINE u8x16 load(const uint8_t* ptr, size_t n, undef<16> src) {
  return internal::partial_load_memcpy(ptr, n, u8x16{});
}
YNN_ALWAYS_INLINE s8x16 load(const int8_t* ptr, size_t n, undef<16> src) {
  return internal::partial_load_memcpy(ptr, n, s8x16{});
}

YNN_ALWAYS_INLINE void store(float* ptr, f32x4 b, size_t n) {
  internal::partial_store_sse(ptr, b, n);
}
YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x4 b, size_t n) {
  internal::partial_store_sse(ptr, b, n);
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
  vec<T, 4> t0{internal::unpacklo_x32x4(x[0].v, x[1].v)};
  vec<T, 4> t1{internal::unpacklo_x32x4(x[2].v, x[3].v)};
  vec<T, 4> t2{internal::unpackhi_x32x4(x[0].v, x[1].v)};
  vec<T, 4> t3{internal::unpackhi_x32x4(x[2].v, x[3].v)};
  return {{
      vec<T, 4>{internal::movelh(t0.v, t1.v)},
      vec<T, 4>{internal::movehl(t1.v, t0.v)},
      vec<T, 4>{internal::movelh(t2.v, t3.v)},
      vec<T, 4>{internal::movehl(t3.v, t2.v)},
  }};
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_SSE2_BASE_H_
