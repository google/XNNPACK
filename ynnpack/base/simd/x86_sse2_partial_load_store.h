// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_SSE2_PARTIAL_LOAD_STORE_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_SSE2_PARTIAL_LOAD_STORE_H_

#include <immintrin.h>

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

namespace internal {

// This implementation of std::copy_n takes a constant upper bound for the
// number of elements.
template <typename T>
YNN_ALWAYS_INLINE void copy_n_small(const T* src, size_t n, T* dst,
                                    std::integral_constant<size_t, 16>) {
  assert(n < 16);
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
  assert(n < 8);
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
  assert(n < 4);
  switch (n) {
    // clang-format off
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
  assert(n < 4);
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

template <typename T>
void partial_store_sse(T* ptr, vec<T, 8> b, size_t n) {
  assert(n < 8);
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
  assert(n < 16);
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

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_SSE2_PARTIAL_LOAD_STORE_H_
