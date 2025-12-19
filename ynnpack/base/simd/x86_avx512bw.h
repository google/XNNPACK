// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_AVX512BW_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_AVX512BW_H_

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/simd/x86_avx512f_base.h"  // IWYU pragma: export

namespace ynn {

namespace simd {

namespace internal {

YNN_ALWAYS_INLINE __m512i mask_loadu(__m512i src, __mmask32 mask,
                                     const bfloat16* ptr) {
  return _mm512_mask_loadu_epi16(src, mask, ptr);
}
YNN_ALWAYS_INLINE __m512i mask_loadu(__m512i src, __mmask32 mask,
                                     const half* ptr) {
  return _mm512_mask_loadu_epi16(src, mask, ptr);
}
YNN_ALWAYS_INLINE __m512i mask_loadu(__m512i src, __mmask32 mask,
                                     const int16_t* ptr) {
  return _mm512_mask_loadu_epi16(src, mask, ptr);
}
YNN_ALWAYS_INLINE __m512i mask_loadu(__m512i src, __mmask64 mask,
                                     const int8_t* ptr) {
  return _mm512_mask_loadu_epi8(src, mask, ptr);
}
YNN_ALWAYS_INLINE __m512i mask_loadu(__m512i src, __mmask64 mask,
                                     const uint8_t* ptr) {
  return _mm512_mask_loadu_epi8(src, mask, ptr);
}

YNN_ALWAYS_INLINE void mask_storeu(half* ptr, __mmask32 mask, __m512i val) {
  _mm512_mask_storeu_epi16(ptr, mask, val);
}
YNN_ALWAYS_INLINE void mask_storeu(bfloat16* ptr, __mmask32 mask, __m512i val) {
  _mm512_mask_storeu_epi16(ptr, mask, val);
}
YNN_ALWAYS_INLINE void mask_storeu(int16_t* ptr, __mmask32 mask, __m512i val) {
  _mm512_mask_storeu_epi16(ptr, mask, val);
}
YNN_ALWAYS_INLINE void mask_storeu(int8_t* ptr, __mmask64 mask, __m512i val) {
  _mm512_mask_storeu_epi8(ptr, mask, val);
}
YNN_ALWAYS_INLINE void mask_storeu(uint8_t* ptr, __mmask64 mask, __m512i val) {
  _mm512_mask_storeu_epi8(ptr, mask, val);
}

// Partial load/store with a non-constant number of elements.
template <typename T>
YNN_ALWAYS_INLINE void partial_store_mask_x16x16(T* ptr, vec<T, 16> val,
                                                 size_t n) {
  assert(n <= 16);
  __mmask16 mask = _cvtu32_mask16((uint32_t)((1 << n) - 1));
  mask_storeu(ptr, mask, val.v);
}

template <typename T>
YNN_ALWAYS_INLINE void partial_store_mask_x16x32(T* ptr, vec<T, 32> val,
                                                 size_t n) {
  assert(n <= 32);
  __mmask32 mask = (1ULL << n) - 1;
  mask_storeu(ptr, mask, val.v);
}

template <typename T>
YNN_ALWAYS_INLINE vec<T, 32> partial_load_mask_x16x32(const T* ptr,
                                                      vec<T, 32> src,
                                                      size_t n) {
  assert(n <= 32);
  __mmask32 mask = (1ULL << n) - 1;
  return vec<T, 32>{mask_loadu(src.v, mask, ptr)};
}

template <typename T>
YNN_ALWAYS_INLINE vec<T, 64> partial_load_mask_x8x64(const T* ptr,
                                                     vec<T, 64> src, size_t n) {
  assert(n < 64);
  __mmask64 mask = _cvtu64_mask64(((1ull << n) - 1));
  return vec<T, 64>{mask_loadu(src.v, mask, ptr)};
}

template <typename T>
YNN_ALWAYS_INLINE void partial_store_mask_x8x64(T* ptr, vec<T, 64> val,
                                                size_t n) {
  assert(n < 64);
  __mmask64 mask = _cvtu64_mask64(((1ull << n) - 1));
  mask_storeu(ptr, mask, val.v);
}

}  // namespace internal

YNN_ALWAYS_INLINE bf16x32 load(const bfloat16* ptr, size_t n, bf16x32 src) {
  return internal::partial_load_mask_x16x32(ptr, src, n);
}
YNN_ALWAYS_INLINE f16x32 load(const half* ptr, size_t n, f16x32 src) {
  return internal::partial_load_mask_x16x32(ptr, src, n);
}
YNN_ALWAYS_INLINE s16x32 load(const int16_t* ptr, size_t n, s16x32 src) {
  return internal::partial_load_mask_x16x32(ptr, src, n);
}
YNN_ALWAYS_INLINE u8x64 load(const uint8_t* ptr, size_t n, u8x64 src) {
  return internal::partial_load_mask_x8x64(ptr, src, n);
}
YNN_ALWAYS_INLINE s8x64 load(const int8_t* ptr, size_t n, s8x64 src) {
  return internal::partial_load_mask_x8x64(ptr, src, n);
}

YNN_ALWAYS_INLINE void store(bfloat16* ptr, bf16x32 val, size_t n) {
  internal::partial_store_mask_x16x32(ptr, val, n);
}
YNN_ALWAYS_INLINE void store(half* ptr, f16x32 val, size_t n) {
  internal::partial_store_mask_x16x32(ptr, val, n);
}
YNN_ALWAYS_INLINE void store(int16_t* ptr, s16x32 val, size_t n) {
  internal::partial_store_mask_x16x32(ptr, val, n);
}
YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x64 val, size_t n) {
  internal::partial_store_mask_x8x64(ptr, val, n);
}
YNN_ALWAYS_INLINE void store(int8_t* ptr, s8x64 val, size_t n) {
  internal::partial_store_mask_x8x64(ptr, val, n);
}

YNN_ALWAYS_INLINE s8x64& operator+=(s8x64& a, s8x64 b) {
  a.v = _mm512_add_epi8(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE u8x64& operator+=(u8x64& a, u8x64 b) {
  a.v = _mm512_add_epi8(a.v, b.v);
  return a;
}

YNN_ALWAYS_INLINE s8x64& operator-=(s8x64& a, s8x64 b) {
  a.v = _mm512_sub_epi8(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE u8x64& operator-=(u8x64& a, u8x64 b) {
  a.v = _mm512_sub_epi8(a.v, b.v);
  return a;
}

YNN_ALWAYS_INLINE s8x64 operator+(s8x64 a, s8x64 b) { return a += b; }
YNN_ALWAYS_INLINE u8x64 operator+(u8x64 a, u8x64 b) { return a += b; }

YNN_ALWAYS_INLINE s8x64 operator-(s8x64 a, s8x64 b) { return a -= b; }
YNN_ALWAYS_INLINE u8x64 operator-(u8x64 a, u8x64 b) { return a -= b; }

YNN_ALWAYS_INLINE s16x32 operator>>(s16x32 a, int b) {
  return s16x32{_mm512_srai_epi16(a.v, b)};
}

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

YNN_ALWAYS_INLINE s32x16 convert(s8x16 a, int32_t) {
  return s32x16{_mm512_cvtepi8_epi32(a.v)};
}
YNN_ALWAYS_INLINE s32x16 convert(u8x16 a, int32_t) {
  return s32x16{_mm512_cvtepu8_epi32(a.v)};
}

using s16x64 = vec<int16_t, 64>;
using s32x32 = vec<int32_t, 32>;
using s32x64 = vec<int32_t, 64>;

YNN_ALWAYS_INLINE s16x32 convert(s8x32 a, int16_t) {
  return s16x32{_mm512_cvtepi8_epi16(a.v)};
}
YNN_ALWAYS_INLINE s16x32 convert(u8x32 a, int16_t) {
  return s16x32{_mm512_cvtepu8_epi16(a.v)};
}

}  // namespace simd

}  // namespace ynn

#include "ynnpack/base/simd/generic.inc"  // IWYU pragma: export

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_AVX512BW_H_
