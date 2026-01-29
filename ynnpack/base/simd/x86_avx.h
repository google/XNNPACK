// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_AVX_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_AVX_H_

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/simd/x86_avx_base.h"  // IWYU pragma: export

namespace ynn {

namespace simd {

// Partial load/store with a non-constant number of elements.
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

YNN_ALWAYS_INLINE f32x4 maskload(const float* ptr, __m128 src, __m128i mask) {
  return f32x4{
      _mm_blendv_ps(src, _mm_maskload_ps(ptr, mask), _mm_castsi128_ps(mask))};
}
YNN_ALWAYS_INLINE s32x4 maskload(const int32_t* ptr, __m128i src,
                                 __m128i mask) {
  return s32x4{_mm_castps_si128(
      _mm_blendv_ps(_mm_castsi128_ps(src),
                    _mm_maskload_ps(reinterpret_cast<const float*>(ptr), mask),
                    _mm_castsi128_ps(mask)))};
}

YNN_ALWAYS_INLINE void maskstore(float* ptr, f32x4 val, __m128i mask) {
  _mm_maskstore_ps(ptr, mask, val.v);
}
YNN_ALWAYS_INLINE void maskstore(int32_t* ptr, s32x4 val, __m128i mask) {
  _mm_maskstore_ps(reinterpret_cast<float*>(ptr), mask,
                   _mm_castsi128_ps(val.v));
}

template <typename T>
YNN_ALWAYS_INLINE vec<T, 8> partial_load_mask_x32x8(const T* ptr, vec<T, 8> src,
                                                    size_t n) {
  assert(n <= 8);
  auto mask =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&mask_table[8 - n]));
  return vec<T, 8>{maskload(ptr, src.v, mask)};
}

template <typename T>
YNN_ALWAYS_INLINE void partial_store_x32x8(T* ptr, vec<T, 8> val, size_t n) {
  assert(n <= 8);
  auto mask =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&mask_table[8 - n]));
  maskstore(ptr, val, mask);
}

template <typename T>
YNN_ALWAYS_INLINE vec<T, 4> partial_load_mask_x32x4(const T* ptr, vec<T, 4> src,
                                                    size_t n) {
  assert(n <= 4);
  auto mask =
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(&mask_table[8 - n]));
  return vec<T, 4>{maskload(ptr, src.v, mask)};
}

template <typename T>
YNN_ALWAYS_INLINE void partial_store_x32x4(T* ptr, vec<T, 4> val, size_t n) {
  assert(n <= 4);
  auto mask =
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(&mask_table[8 - n]));
  maskstore(ptr, val, mask);
}

}  // namespace internal

YNN_ALWAYS_INLINE f32x8 load(const float* ptr, size_t n, f32x8 src) {
  return internal::partial_load_mask_x32x8(ptr, src, n);
}
YNN_ALWAYS_INLINE s32x8 load(const int32_t* ptr, size_t n, s32x8 src) {
  return internal::partial_load_mask_x32x8(ptr, src, n);
}
YNN_ALWAYS_INLINE bf16x16 load(const bfloat16* ptr, size_t n, bf16x16 src) {
  return internal::partial_load_memcpy(ptr, src, n);
}
YNN_ALWAYS_INLINE f16x16 load(const half* ptr, size_t n, f16x16 src) {
  return internal::partial_load_memcpy(ptr, src, n);
}
YNN_ALWAYS_INLINE s16x16 load(const int16_t* ptr, size_t n, s16x16 src) {
  return internal::partial_load_memcpy(ptr, src, n);
}
YNN_ALWAYS_INLINE u8x32 load(const uint8_t* ptr, size_t n, u8x32 src) {
  return internal::partial_load_memcpy(ptr, src, n);
}
YNN_ALWAYS_INLINE s8x32 load(const int8_t* ptr, size_t n, s8x32 src) {
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

YNN_ALWAYS_INLINE f32x4 load(const float* ptr, size_t n, f32x4 src) {
  return internal::partial_load_mask_x32x4(ptr, src, n);
}
YNN_ALWAYS_INLINE s32x4 load(const int32_t* ptr, size_t n, s32x4 src) {
  return internal::partial_load_mask_x32x4(ptr, src, n);
}
YNN_ALWAYS_INLINE bf16x8 load(const bfloat16* ptr, size_t n, bf16x8 src) {
  return internal::partial_load_memcpy(ptr, src, n);
}
YNN_ALWAYS_INLINE f16x8 load(const half* ptr, size_t n, f16x8 src) {
  return internal::partial_load_memcpy(ptr, src, n);
}
YNN_ALWAYS_INLINE s16x8 load(const int16_t* ptr, size_t n, s16x8 src) {
  return internal::partial_load_memcpy(ptr, src, n);
}
YNN_ALWAYS_INLINE u8x16 load(const uint8_t* ptr, size_t n, u8x16 src) {
  return internal::partial_load_memcpy(ptr, src, n);
}
YNN_ALWAYS_INLINE s8x16 load(const int8_t* ptr, size_t n, s8x16 src) {
  return internal::partial_load_memcpy(ptr, src, n);
}

YNN_ALWAYS_INLINE void store(float* ptr, f32x4 val, size_t n) {
  internal::partial_store_x32x4(ptr, val, n);
}
YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x4 val, size_t n) {
  internal::partial_store_x32x4(ptr, val, n);
}
YNN_ALWAYS_INLINE void store(bfloat16* ptr, bf16x8 val, size_t n) {
  internal::partial_store_memcpy(ptr, val, n);
}
YNN_ALWAYS_INLINE void store(half* ptr, f16x8 val, size_t n) {
  internal::partial_store_memcpy(ptr, val, n);
}
YNN_ALWAYS_INLINE void store(int16_t* ptr, s16x8 val, size_t n) {
  internal::partial_store_memcpy(ptr, val, n);
}
YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x16 val, size_t n) {
  internal::partial_store_memcpy(ptr, val, n);
}
YNN_ALWAYS_INLINE void store(int8_t* ptr, s8x16 val, size_t n) {
  internal::partial_store_memcpy(ptr, val, n);
}

using f32x16 = vec<float, 16>;
using s32x16 = vec<int32_t, 16>;
using bf16x32 = vec<bfloat16, 32>;
using f16x32 = vec<half, 32>;
using s8x64 = vec<int8_t, 64>;
using u8x64 = vec<uint8_t, 64>;

}  // namespace simd

}  // namespace ynn

#include "ynnpack/base/simd/generic.inc"  // IWYU pragma: export

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_AVX_H_
