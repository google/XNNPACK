// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_AVX_MASK_LOAD_STORE_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_AVX_MASK_LOAD_STORE_H_

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/simd/x86_avx_base.h"  // IWYU pragma: export

namespace ynn {

namespace simd {

namespace internal {

// Align this to avoid spanning a cache line.
alignas(64) static constexpr int32_t mask_table[16] = {
    -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};

YNN_ALWAYS_INLINE f32x8 maskload(const float* ptr, __m256i mask) {
  return f32x8{_mm256_maskload_ps(ptr, mask)};
}
YNN_ALWAYS_INLINE s32x8 maskload(const int32_t* ptr, __m256i mask) {
  return s32x8{_mm256_castps_si256(
      _mm256_maskload_ps(reinterpret_cast<const float*>(ptr), mask))};
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

YNN_ALWAYS_INLINE void maskstore(float* ptr, f32x8 val, __m256i mask) {
  _mm256_maskstore_ps(ptr, mask, val.v);
}
YNN_ALWAYS_INLINE void maskstore(int32_t* ptr, s32x8 val, __m256i mask) {
  _mm256_maskstore_ps(reinterpret_cast<float*>(ptr), mask,
                      _mm256_castsi256_ps(val.v));
}

// Partial load/store with a non-constant number of elements.
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
YNN_ALWAYS_INLINE void partial_store_x32x8(T* ptr, vec<T, 8> val, size_t n) {
  assert(n <= 8);
  auto mask =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&mask_table[8 - n]));
  maskstore(ptr, val, mask);
}

}  // namespace internal

YNN_ALWAYS_INLINE f32x8 load(const float* ptr, size_t n, f32x8 src) {
  return internal::partial_load_mask_x32x8(ptr, n, src);
}
YNN_ALWAYS_INLINE s32x8 load(const int32_t* ptr, size_t n, s32x8 src) {
  return internal::partial_load_mask_x32x8(ptr, n, src);
}

YNN_ALWAYS_INLINE f32x8 load(const float* ptr, size_t n, zeros<8> src) {
  return internal::partial_load_mask_x32x8(ptr, n, src);
}
YNN_ALWAYS_INLINE s32x8 load(const int32_t* ptr, size_t n, zeros<8> src) {
  return internal::partial_load_mask_x32x8(ptr, n, src);
}

YNN_ALWAYS_INLINE f32x8 load(const float* ptr, size_t n, undef<8> src) {
  return internal::partial_load_mask_x32x8(ptr, n, zeros<8>{});
}
YNN_ALWAYS_INLINE s32x8 load(const int32_t* ptr, size_t n, undef<8> src) {
  return internal::partial_load_mask_x32x8(ptr, n, zeros<8>{});
}

YNN_ALWAYS_INLINE void store(float* ptr, f32x8 val, size_t n) {
  internal::partial_store_x32x8(ptr, val, n);
}
YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x8 val, size_t n) {
  internal::partial_store_x32x8(ptr, val, n);
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_AVX_MASK_LOAD_STORE_H_
