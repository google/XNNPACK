// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_AVX512F_BASE_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_AVX512F_BASE_H_

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
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

YNN_ALWAYS_INLINE __m256i lo(__m512i x) { return _mm512_castsi512_si256(x); }
YNN_ALWAYS_INLINE __m256i hi(__m512i x) {
  return _mm512_extracti64x4_epi64(x, 1);
}

}  // namespace internal

template <>
struct vec<float, 16> {
  using value_type = float;
  static constexpr std::integral_constant<size_t, 16> N = {};

  vec() = default;
  explicit vec(__m512 v) : v(v) {}
  vec(float x) : v(_mm512_set1_ps(x)) {}  // NOLINT

  __m512 v;

  YNN_ALWAYS_INLINE f32x8 lo() const { return f32x8{internal::lo(v)}; }
  YNN_ALWAYS_INLINE f32x8 hi() const { return f32x8{internal::hi(v)}; }
};

template <>
struct vec<int32_t, 16> {
  using value_type = int32_t;
  static constexpr std::integral_constant<size_t, 16> N = {};

  vec() = default;
  explicit vec(__m512i v) : v(v) {}
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
  vec(half x) : v(_mm512_set1_epi16(x.to_bits())) {}  // NOLINT

  __m512i v;

  YNN_ALWAYS_INLINE f16x16 lo() const { return f16x16{internal::lo(v)}; }
  YNN_ALWAYS_INLINE f16x16 hi() const { return f16x16{internal::hi(v)}; }
};

template <>
struct vec<int16_t, 32> {
  using value_type = int16_t;
  static constexpr std::integral_constant<size_t, 32> N = {};

  vec() = default;
  explicit vec(__m512i v) : v(v) {}
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
  vec(int8_t x) : v(_mm512_set1_epi8(x)) {}  // NOLINT

  __m512i v;

  YNN_ALWAYS_INLINE s8x32 lo() const { return s8x32{internal::lo(v)}; }
  YNN_ALWAYS_INLINE s8x32 hi() const { return s8x32{internal::hi(v)}; }
};

using f32x16 = vec<float, 16>;
using s32x16 = vec<int32_t, 16>;
using bf16x32 = vec<bfloat16, 32>;
using f16x32 = vec<half, 32>;
using s16x32 = vec<int16_t, 32>;
using u8x64 = vec<uint8_t, 64>;
using s8x64 = vec<int8_t, 64>;

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

YNN_ALWAYS_INLINE void store_aligned(float* ptr, f32x16 b,
                                     decltype(f32x16::N) = {}) {
  _mm512_store_ps(ptr, b.v);
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

YNN_ALWAYS_INLINE void store(float* ptr, f32x16 b, decltype(f32x16::N) = {}) {
  _mm512_storeu_ps(ptr, b.v);
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

YNN_ALWAYS_INLINE __m512 mask_loadu(__m512 src, __mmask16 mask,
                                    const float* ptr) {
  return _mm512_mask_loadu_ps(src, mask, ptr);
}
YNN_ALWAYS_INLINE __m512i mask_loadu(__m512i src, __mmask16 mask,
                                     const int32_t* ptr) {
  return _mm512_mask_loadu_epi32(src, mask, ptr);
}

YNN_ALWAYS_INLINE void mask_storeu(float* ptr, __mmask16 mask, __m512 val) {
  _mm512_mask_storeu_ps(ptr, mask, val);
}
YNN_ALWAYS_INLINE void mask_storeu(int32_t* ptr, __mmask16 mask, __m512i val) {
  _mm512_mask_storeu_epi32(ptr, mask, val);
}

// Partial load/store with a non-constant number of elements.
template <typename T>
YNN_ALWAYS_INLINE vec<T, 16> partial_load_mask_x32x16(const T* ptr,
                                                      vec<T, 16> src,
                                                      size_t n) {
  assert(n <= 16);
  __mmask16 mask = _cvtu32_mask16((uint32_t)((1 << n) - 1));
  return vec<T, 16>{mask_loadu(src.v, mask, ptr)};
}

template <typename T>
YNN_ALWAYS_INLINE void partial_store_mask_x32x16(T* ptr, vec<T, 16> val,
                                                 size_t n) {
  assert(n <= 16);
  __mmask16 mask = _cvtu32_mask16((uint32_t)((1 << n) - 1));
  mask_storeu(ptr, mask, val.v);
}

}  // namespace internal

YNN_ALWAYS_INLINE f32x16 load(const float* ptr, size_t n, f32x16 src) {
  return internal::partial_load_mask_x32x16(ptr, src, n);
}
YNN_ALWAYS_INLINE s32x16 load(const int32_t* ptr, size_t n, s32x16 src) {
  return internal::partial_load_mask_x32x16(ptr, src, n);
}

YNN_ALWAYS_INLINE void store(float* ptr, f32x16 val, size_t n) {
  internal::partial_store_mask_x32x16(ptr, val, n);
}
YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x16 val, size_t n) {
  internal::partial_store_mask_x32x16(ptr, val, n);
}

YNN_ALWAYS_INLINE f32x16& operator+=(f32x16& a, f32x16 b) {
  a.v = _mm512_add_ps(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s32x16& operator+=(s32x16& a, s32x16 b) {
  a.v = _mm512_add_epi32(a.v, b.v);
  return a;
}

YNN_ALWAYS_INLINE f32x16& operator-=(f32x16& a, f32x16 b) {
  a.v = _mm512_sub_ps(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s32x16& operator-=(s32x16& a, s32x16 b) {
  a.v = _mm512_sub_epi32(a.v, b.v);
  return a;
}

YNN_ALWAYS_INLINE f32x16& operator*=(f32x16& a, f32x16 b) {
  a.v = _mm512_mul_ps(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s32x16& operator*=(s32x16& a, s32x16 b) {
  a.v = _mm512_mullo_epi32(a.v, b.v);
  return a;
}

YNN_ALWAYS_INLINE f32x16 operator+(f32x16 a, f32x16 b) { return a += b; }
YNN_ALWAYS_INLINE s32x16 operator+(s32x16 a, s32x16 b) { return a += b; }

YNN_ALWAYS_INLINE f32x16 operator-(f32x16 a, f32x16 b) { return a -= b; }
YNN_ALWAYS_INLINE s32x16 operator-(s32x16 a, s32x16 b) { return a -= b; }

YNN_ALWAYS_INLINE f32x16 operator*(f32x16 a, f32x16 b) { return a *= b; }
YNN_ALWAYS_INLINE s32x16 operator*(s32x16 a, s32x16 b) { return a *= b; }

YNN_ALWAYS_INLINE s16x32 operator&(s16x32 a, int b) {
  return s16x32{_mm512_and_si512(a.v, _mm512_set1_epi16(b))};
}
YNN_ALWAYS_INLINE s16x32 operator^(s16x32 a, s16x32 b) {
  return s16x32{_mm512_xor_si512(a.v, b.v)};
}

YNN_ALWAYS_INLINE f32x16 fma(f32x16 a, f32x16 b, f32x16 acc) {
  return f32x16{_mm512_fmadd_ps(a.v, b.v, acc.v)};
}

YNN_ALWAYS_INLINE f32x16 min(f32x16 a, f32x16 b) {
  return f32x16{_mm512_min_ps(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x16 min(s32x16 a, s32x16 b) {
  return s32x16{_mm512_min_epi32(a.v, b.v)};
}

YNN_ALWAYS_INLINE f32x16 max(f32x16 a, f32x16 b) {
  return f32x16{_mm512_max_ps(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x16 max(s32x16 a, s32x16 b) {
  return s32x16{_mm512_max_epi32(a.v, b.v)};
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

YNN_ALWAYS_INLINE f32x16 concat(f32x8 x, f32x8 y) {
  return f32x16{_mm512_castpd_ps(
      _mm512_insertf64x4(_mm512_castps_pd(_mm512_castps256_ps512(x.v)),
                         _mm256_castps_pd(y.v), 1))};
}
YNN_ALWAYS_INLINE s32x16 concat(s32x8 x, s32x8 y) {
  return s32x16{_mm512_inserti64x4(_mm512_castsi256_si512(x.v), y.v, 1)};
}
YNN_ALWAYS_INLINE bf16x32 concat(bf16x16 x, bf16x16 y) {
  return bf16x32{_mm512_inserti64x4(_mm512_castsi256_si512(x.v), y.v, 1)};
}
YNN_ALWAYS_INLINE f16x32 concat(f16x16 x, f16x16 y) {
  return f16x32{_mm512_inserti64x4(_mm512_castsi256_si512(x.v), y.v, 1)};
}
YNN_ALWAYS_INLINE s8x64 concat(s8x32 x, s8x32 y) {
  return s8x64{_mm512_inserti64x4(_mm512_castsi256_si512(x.v), y.v, 1)};
}
YNN_ALWAYS_INLINE u8x64 concat(u8x32 x, u8x32 y) {
  return u8x64{_mm512_inserti64x4(_mm512_castsi256_si512(x.v), y.v, 1)};
}

using f32x32 = vec<float, 32>;

YNN_ALWAYS_INLINE f32x16 convert(f16x16 x, float) {
  return f32x16{_mm512_cvtph_ps(x.v)};
}

YNN_ALWAYS_INLINE f32x16 convert(bf16x16 a, float) {
  return f32x16{_mm512_castsi512_ps(_mm512_slli_epi32(
      _mm512_cvtepu16_epi32(a.v), 16))};
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_AVX512F_H_
