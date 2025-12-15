// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_AVX512F_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_AVX512F_H_

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/multi_vec.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/simd/x86_avx2_base.h"  // IWYU pragma: export

namespace ynn {

namespace simd {

// See vec.h for architecture independent comments.

template <>
struct vec<float, 16> {
  using value_type = float;
  static constexpr std::integral_constant<size_t, 16> N = {};

  vec() = default;
  explicit vec(__m512 v) : v(v) {}
  vec(float x) : v(_mm512_set1_ps(x)) {}  // NOLINT

  __m512 v;
};

template <>
struct vec<int32_t, 16> {
  using value_type = int32_t;
  static constexpr std::integral_constant<size_t, 16> N = {};

  vec() = default;
  explicit vec(__m512i v) : v(v) {}
  vec(int32_t x) : v(_mm512_set1_epi32(x)) {}  // NOLINT

  __m512i v;
};

template <>
struct vec<bfloat16, 32> {
  using value_type = bfloat16;
  static constexpr std::integral_constant<size_t, 32> N = {};

  vec() = default;
  explicit vec(__m512i v) : v(v) {}
  vec(bfloat16 x) : v(_mm512_set1_epi16(x.to_bits())) {}  // NOLINT

  __m512i v;
};

template <>
struct vec<half, 32> {
  using value_type = half;
  static constexpr std::integral_constant<size_t, 32> N = {};

  vec() = default;
  explicit vec(__m512i v) : v(v) {}
  vec(half x) : v(_mm512_set1_epi16(x.to_bits())) {}  // NOLINT

  __m512i v;
};

template <>
struct vec<int16_t, 32> {
  using value_type = int16_t;
  static constexpr std::integral_constant<size_t, 32> N = {};

  vec() = default;
  explicit vec(__m512i v) : v(v) {}
  vec(int16_t x) : v(_mm512_set1_epi16(x)) {}  // NOLINT

  __m512i v;
};

template <>
struct vec<uint8_t, 64> {
  using value_type = uint8_t;
  static constexpr std::integral_constant<size_t, 64> N = {};

  vec() = default;
  explicit vec(__m512i v) : v(v) {}
  vec(uint8_t x) : v(_mm512_set1_epi8(x)) {}  // NOLINT

  __m512i v;
};

template <>
struct vec<int8_t, 64> {
  using value_type = int8_t;
  static constexpr std::integral_constant<size_t, 64> N = {};

  vec() = default;
  explicit vec(__m512i v) : v(v) {}
  vec(int8_t x) : v(_mm512_set1_epi8(x)) {}  // NOLINT

  __m512i v;
};

using f32x16 = vec<float, 16>;
using s32x16 = vec<int32_t, 16>;
using bf16x32 = vec<bfloat16, 32>;
using f16x32 = vec<half, 32>;
using s16x32 = vec<int16_t, 32>;
using u8x64 = vec<uint8_t, 64>;
using s8x64 = vec<int8_t, 64>;

YNN_ALWAYS_INLINE f32x16 load_aligned(const float* ptr, f32x16,
                                      decltype(f32x16::N) = {}) {
  return f32x16{_mm512_load_ps(ptr)};
}
YNN_ALWAYS_INLINE s32x16 load_aligned(const int32_t* ptr, s32x16,
                                      decltype(s32x16::N) = {}) {
  return s32x16{_mm512_load_si512(reinterpret_cast<const __m512i*>(ptr))};
}
YNN_ALWAYS_INLINE bf16x32 load_aligned(const bfloat16* ptr, bf16x32,
                                       decltype(bf16x32::N) = {}) {
  return bf16x32{_mm512_load_si512(reinterpret_cast<const __m512i*>(ptr))};
}
YNN_ALWAYS_INLINE f16x32 load_aligned(const half* ptr, f16x32,
                                      decltype(f16x32::N) = {}) {
  return f16x32{_mm512_load_si512(reinterpret_cast<const __m512i*>(ptr))};
}
YNN_ALWAYS_INLINE s16x32 load_aligned(const int16_t* ptr, s16x32,
                                      decltype(s16x32::N) = {}) {
  return s16x32{_mm512_load_si512(reinterpret_cast<const __m512i*>(ptr))};
}
YNN_ALWAYS_INLINE u8x64 load_aligned(const uint8_t* ptr, u8x64,
                                     decltype(u8x64::N) = {}) {
  return u8x64{_mm512_load_si512(reinterpret_cast<const __m512i*>(ptr))};
}
YNN_ALWAYS_INLINE s8x64 load_aligned(const int8_t* ptr, s8x64,
                                     decltype(s8x64::N) = {}) {
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

YNN_ALWAYS_INLINE f32x16 load(const float* ptr, f32x16,
                              decltype(f32x16::N) = {}) {
  return f32x16{_mm512_loadu_ps(ptr)};
}
YNN_ALWAYS_INLINE s32x16 load(const int32_t* ptr, s32x16,
                              decltype(s32x16::N) = {}) {
  return s32x16{_mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr))};
}
YNN_ALWAYS_INLINE bf16x32 load(const bfloat16* ptr, bf16x32,
                               decltype(bf16x32::N) = {}) {
  return bf16x32{_mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr))};
}
YNN_ALWAYS_INLINE f16x32 load(const half* ptr, f16x32,
                              decltype(f16x32::N) = {}) {
  return f16x32{_mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr))};
}
YNN_ALWAYS_INLINE s16x32 load(const int16_t* ptr, s16x32,
                              decltype(s16x32::N) = {}) {
  return s16x32{_mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr))};
}
YNN_ALWAYS_INLINE u8x64 load(const uint8_t* ptr, u8x64,
                             decltype(u8x64::N) = {}) {
  return u8x64{_mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr))};
}
YNN_ALWAYS_INLINE s8x64 load(const int8_t* ptr, s8x64,
                             decltype(s8x64::N) = {}) {
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
inline vec<T, 16> partial_load_mask_x32x16(const T* ptr, vec<T, 16> src,
                                           size_t n) {
  assert(n <= 16);
  __mmask16 mask = _cvtu32_mask16((uint32_t)((1 << n) - 1));
  return vec<T, 16>{mask_loadu(src.v, mask, ptr)};
}

template <typename T>
inline void partial_store_mask_x32x16(T* ptr, vec<T, 16> val, size_t n) {
  assert(n <= 16);
  __mmask16 mask = _cvtu32_mask16((uint32_t)((1 << n) - 1));
  mask_storeu(ptr, mask, val.v);
}

}  // namespace internal

YNN_ALWAYS_INLINE f32x16 load(const float* ptr, f32x16 src, size_t n) {
  return internal::partial_load_mask_x32x16(ptr, src, n);
}
YNN_ALWAYS_INLINE s32x16 load(const int32_t* ptr, s32x16 src, size_t n) {
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

YNN_ALWAYS_INLINE float horizontal_max(f32x16 a) {
  const __m512 swapped = _mm512_shuffle_f32x4(a.v, a.v, 0x4E);
  const __m512 max512 = _mm512_max_ps(a.v, swapped);
  const __m256 max256 = _mm512_castps512_ps256(max512);
  const __m128 max128 = _mm_max_ps(_mm256_castps256_ps128(max256),
                                   _mm256_extractf128_ps(max256, 1));

  const __m128 max_pairs = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
  return _mm_cvtss_f32(
      _mm_max_ss(max_pairs, _mm_shuffle_ps(max_pairs, max_pairs, 1)));
}
YNN_ALWAYS_INLINE float horizontal_min(f32x16 a) {
  const __m512 swapped = _mm512_shuffle_f32x4(a.v, a.v, 0x4E);
  const __m512 min512 = _mm512_min_ps(a.v, swapped);
  const __m256 min256 = _mm512_castps512_ps256(min512);
  const __m128 min128 = _mm_min_ps(_mm256_castps256_ps128(min256),
                                   _mm256_extractf128_ps(min256, 1));

  const __m128 min_pairs = _mm_min_ps(min128, _mm_movehl_ps(min128, min128));
  return _mm_cvtss_f32(
      _mm_min_ss(min_pairs, _mm_shuffle_ps(min_pairs, min_pairs, 1)));
}

YNN_ALWAYS_INLINE int32_t horizontal_max(s32x16 a) {
  const __m256i max16 = _mm256_max_epi32(_mm512_castsi512_si256(a.v),
                                         _mm512_extracti64x4_epi64(a.v, 1));
  const __m128i max8 = _mm_max_epi32(_mm256_castsi256_si128(max16),
                                     _mm256_extracti128_si256(max16, 1));
  const __m128i max4 = _mm_max_epi32(max8, _mm_srli_si128(max8, 8));
  return _mm_cvtsi128_si32(_mm_max_epi32(max4, _mm_srli_si128(max4, 4)));
}
YNN_ALWAYS_INLINE int32_t horizontal_min(s32x16 a) {
  const __m256i min16 = _mm256_min_epi32(_mm512_castsi512_si256(a.v),
                                         _mm512_extracti64x4_epi64(a.v, 1));
  const __m128i min8 = _mm_min_epi32(_mm256_castsi256_si128(min16),
                                     _mm256_extracti128_si256(min16, 1));
  const __m128i min4 = _mm_min_epi32(min8, _mm_srli_si128(min8, 8));
  return _mm_cvtsi128_si32(_mm_min_epi32(min4, _mm_srli_si128(min4, 4)));
}

YNN_ALWAYS_INLINE int16_t horizontal_max(s16x32 a) {
  const __m256i max16 = _mm256_max_epi16(_mm512_castsi512_si256(a.v),
                                         _mm512_extracti64x4_epi64(a.v, 1));
  const __m128i max8 = _mm_max_epi16(_mm256_castsi256_si128(max16),
                                     _mm256_extracti128_si256(max16, 1));
  const __m128i max4 = _mm_max_epi16(max8, _mm_srli_si128(max8, 8));
  const __m128i max2 = _mm_max_epi16(max4, _mm_srli_si128(max4, 4));
  return static_cast<int16_t>(
      _mm_cvtsi128_si32(_mm_max_epi16(max2, _mm_srli_si128(max2, 2))));
}
YNN_ALWAYS_INLINE int16_t horizontal_min(s16x32 a) {
  const __m256i min16 = _mm256_min_epi16(_mm512_castsi512_si256(a.v),
                                         _mm512_extracti64x4_epi64(a.v, 1));
  const __m128i min8 = _mm_min_epi16(_mm256_castsi256_si128(min16),
                                     _mm256_extracti128_si256(min16, 1));
  const __m128i min4 = _mm_min_epi16(min8, _mm_srli_si128(min8, 8));
  const __m128i min2 = _mm_min_epi16(min4, _mm_srli_si128(min4, 4));
  return static_cast<int16_t>(
      _mm_cvtsi128_si32(_mm_min_epi16(min2, _mm_srli_si128(min2, 2))));
}

YNN_ALWAYS_INLINE int8_t horizontal_max(s8x64 a) {
  const __m256i low256 = _mm512_castsi512_si256(a.v);
  const __m256i high256 = _mm512_extracti64x4_epi64(a.v, 1);
  const __m256i max32 = _mm256_max_epi8(low256, high256);

  const __m128i low128 = _mm256_castsi256_si128(max32);
  const __m128i high128 = _mm256_extracti128_si256(max32, 1);
  const __m128i max16 = _mm_max_epi8(low128, high128);

  const __m128i max8 = _mm_max_epi8(max16, _mm_srli_si128(max16, 8));
  const __m128i max4 = _mm_max_epi8(max8, _mm_srli_si128(max8, 4));
  const __m128i max2 = _mm_max_epi8(max4, _mm_srli_si128(max4, 2));

  return static_cast<int8_t>(
      _mm_cvtsi128_si32(_mm_max_epi8(max2, _mm_srli_si128(max2, 1))));
}
YNN_ALWAYS_INLINE int8_t horizontal_min(s8x64 a) {
  const __m256i low256 = _mm512_castsi512_si256(a.v);
  const __m256i high256 = _mm512_extracti64x4_epi64(a.v, 1);
  const __m256i min32 = _mm256_min_epi8(low256, high256);

  const __m128i low128 = _mm256_castsi256_si128(min32);
  const __m128i high128 = _mm256_extracti128_si256(min32, 1);
  const __m128i min16 = _mm_min_epi8(low128, high128);

  const __m128i min8 = _mm_min_epi8(min16, _mm_srli_si128(min16, 8));
  const __m128i min4 = _mm_min_epi8(min8, _mm_srli_si128(min8, 4));
  const __m128i min2 = _mm_min_epi8(min4, _mm_srli_si128(min4, 2));

  return static_cast<int8_t>(
      _mm_cvtsi128_si32(_mm_min_epi8(min2, _mm_srli_si128(min2, 1))));
}

YNN_ALWAYS_INLINE uint8_t horizontal_max(u8x64 a) {
  const __m256i low256 = _mm512_castsi512_si256(a.v);
  const __m256i high256 = _mm512_extracti64x4_epi64(a.v, 1);
  const __m256i max32 = _mm256_max_epu8(low256, high256);

  const __m128i low128 = _mm256_castsi256_si128(max32);
  const __m128i high128 = _mm256_extracti128_si256(max32, 1);
  const __m128i max16 = _mm_max_epu8(low128, high128);

  const __m128i max8 = _mm_max_epu8(max16, _mm_srli_si128(max16, 8));
  const __m128i max4 = _mm_max_epu8(max8, _mm_srli_si128(max8, 4));
  const __m128i max2 = _mm_max_epu8(max4, _mm_srli_si128(max4, 2));

  return static_cast<uint8_t>(
      _mm_cvtsi128_si32(_mm_max_epu8(max2, _mm_srli_si128(max2, 1))));
}
YNN_ALWAYS_INLINE uint8_t horizontal_min(u8x64 a) {
  const __m256i low256 = _mm512_castsi512_si256(a.v);
  const __m256i high256 = _mm512_extracti64x4_epi64(a.v, 1);
  const __m256i min32 = _mm256_min_epu8(low256, high256);

  const __m128i low128 = _mm256_castsi256_si128(min32);
  const __m128i high128 = _mm256_extracti128_si256(min32, 1);
  const __m128i min16 = _mm_min_epu8(low128, high128);

  const __m128i min8 = _mm_min_epu8(min16, _mm_srli_si128(min16, 8));
  const __m128i min4 = _mm_min_epu8(min8, _mm_srli_si128(min8, 4));
  const __m128i min2 = _mm_min_epu8(min4, _mm_srli_si128(min4, 2));

  return static_cast<uint8_t>(
      _mm_cvtsi128_si32(_mm_min_epu8(min2, _mm_srli_si128(min2, 1))));
}

template <int Index>
YNN_ALWAYS_INLINE s32x8 extract(s32x16 x, s32x8) {
  assert(Index == 0);
  return s32x8{_mm512_castsi512_si256(x.v)};
}
template <int Index>
YNN_ALWAYS_INLINE s8x32 extract(s8x64 x, s8x32) {
  return s8x32{_mm512_extracti64x4_epi64(x.v, Index)};
}
template <int Index>
YNN_ALWAYS_INLINE u8x32 extract(u8x64 x, u8x32) {
  return u8x32{_mm512_extracti64x4_epi64(x.v, Index)};
}

template <int Index>
YNN_ALWAYS_INLINE s32x4 extract(s32x16 x, s32x4) {
  return s32x4{_mm512_extracti32x4_epi32(x.v, Index)};
}
template <int Index>
YNN_ALWAYS_INLINE f32x4 extract(f32x16 x, f32x4) {
  return f32x4{_mm512_extractf32x4_ps(x.v, Index)};
}
template <int Index>
YNN_ALWAYS_INLINE s8x16 extract(s8x64 x, s8x16) {
  return s8x16{_mm512_extracti32x4_epi32(x.v, Index)};
}
template <int Index>
YNN_ALWAYS_INLINE u8x16 extract(u8x64 x, u8x16) {
  return u8x16{_mm512_extracti32x4_epi32(x.v, Index)};
}

template <int Index>
YNN_ALWAYS_INLINE bf16x16 extract(bf16x32 x, bf16x16) {
  return bf16x16{_mm256_castpd_si256(
      _mm512_extractf64x4_pd(_mm512_castsi512_pd(x.v), Index))};
}
template <int Index>
YNN_ALWAYS_INLINE f16x16 extract(f16x32 x, f16x16) {
  return f16x16{_mm256_castpd_si256(
      _mm512_extractf64x4_pd(_mm512_castsi512_pd(x.v), Index))};
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

using f32x32 = multi_vec<f32x16, 2>;

YNN_ALWAYS_INLINE f32x16 convert(f16x16 x, float) {
  return f32x16{_mm512_cvtph_ps(x.v)};
}

YNN_ALWAYS_INLINE f32x16 convert(bf16x16 a, float) {
  return f32x16{_mm512_castsi512_ps(_mm512_slli_epi32(
      _mm512_cvtepu16_epi32(a.v), 16))};
}

YNN_ALWAYS_INLINE f32x32 convert(f16x32 a, float) {
  return {
      convert(extract<0>(a, f16x16{}), float{}),
      convert(extract<1>(a, f16x16{}), float{}),
  };
}

YNN_ALWAYS_INLINE f32x32 convert(bf16x32 a, float) {
  return {
      convert(extract<0>(a, bf16x16{}), float{}),
      convert(extract<1>(a, bf16x16{}), float{}),
  };
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_AVX512F_H_
