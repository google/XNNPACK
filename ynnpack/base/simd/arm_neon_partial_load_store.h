// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_ARM_NEON_PARTIAL_LOAD_STORE_H_
#define XNNPACK_YNNPACK_BASE_SIMD_ARM_NEON_PARTIAL_LOAD_STORE_H_

#include <arm_neon.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>

#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/arm_neon_base.h"
#include "ynnpack/base/simd/vec.h"

namespace ynn {

namespace simd {

namespace internal {

template <typename T, size_t N>
void store_aligned(T* dst, zeros<N>) {
  memset(dst, 0, N * sizeof(T));
}

template <typename T, size_t N>
void store_aligned(T* dst, undef<N>) {}

// Partial load/store with a non-constant number of elements.
template <typename T, typename Init>
inline vec<T, 4> partial_load_neon(const T* ptr, size_t n, Init src) {
  assert(n <= 4);
  if (n == 4) {
    return load(ptr, std::integral_constant<size_t, 4>{});
  }
  alignas(vec<T, 4>) T lanes[4];
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
  return load_aligned(lanes, std::integral_constant<size_t, 4>{});
}
template <typename T>
inline void partial_store_neon(T* ptr, vec<T, 4> b, size_t n) {
  assert(n <= 4);
  if (n == 4) {
    store(ptr, b);
    return;
  }
  switch (n) {
    case 3:
      vst1_lane<0>(&ptr[2], vget_high(b.v));
      [[fallthrough]];
    case 2:
      vst1_lane<1>(&ptr[1], vget_low(b.v));
      [[fallthrough]];
    case 1:
      vst1_lane<0>(&ptr[0], vget_low(b.v));
      break;
    default:
      break;
  }
}

}  // namespace internal

YNN_ALWAYS_INLINE f32x4 load(const float* ptr, size_t n, f32x4 src) {
  return internal::partial_load_neon(ptr, n, src);
}
YNN_ALWAYS_INLINE s32x4 load(const int32_t* ptr, size_t n, s32x4 src) {
  return internal::partial_load_neon(ptr, n, src);
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

YNN_ALWAYS_INLINE f16x4 load(const half* ptr, size_t n, f16x4 src) {
  return internal::partial_load_memcpy(ptr, n, src);
}
YNN_ALWAYS_INLINE u8x8 load(const uint8_t* ptr, size_t n, u8x8 src) {
  return internal::partial_load_memcpy(ptr, n, src);
}

YNN_ALWAYS_INLINE f32x4 load(const float* ptr, size_t n, zeros<4> src) {
  return internal::partial_load_neon(ptr, n, src);
}
YNN_ALWAYS_INLINE s32x4 load(const int32_t* ptr, size_t n, zeros<4> src) {
  return internal::partial_load_neon(ptr, n, src);
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

YNN_ALWAYS_INLINE f16x4 load(const half* ptr, size_t n, zeros<4> src) {
  return internal::partial_load_memcpy(ptr, n, f16x4{0});
}
YNN_ALWAYS_INLINE u8x8 load(const uint8_t* ptr, size_t n, zeros<8> src) {
  return internal::partial_load_memcpy(ptr, n, u8x8{0});
}

YNN_ALWAYS_INLINE f32x4 load(const float* ptr, size_t n, undef<4> src) {
  return internal::partial_load_neon(ptr, n, src);
}
YNN_ALWAYS_INLINE s32x4 load(const int32_t* ptr, size_t n, undef<4> src) {
  return internal::partial_load_neon(ptr, n, src);
}
YNN_ALWAYS_INLINE bf16x8 load(const bfloat16* ptr, size_t n, undef<8> src) {
  return internal::partial_load_memcpy(ptr, n, bf16x8{});
}
YNN_ALWAYS_INLINE f16x4 load(const half* ptr, size_t n, undef<4> src) {
  return internal::partial_load_memcpy(ptr, n, f16x4{});
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

YNN_ALWAYS_INLINE u8x8 load(const uint8_t* ptr, size_t n, undef<8> src) {
  return internal::partial_load_memcpy(ptr, n, u8x8{});
}

YNN_ALWAYS_INLINE void store(float* ptr, f32x4 b, size_t n) {
  internal::partial_store_neon(ptr, b, n);
}
YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x4 b, size_t n) {
  internal::partial_store_neon(ptr, b, n);
}
YNN_ALWAYS_INLINE void store(bfloat16* ptr, bf16x8 value, size_t n) {
  internal::partial_store_memcpy(ptr, value, n);
}
YNN_ALWAYS_INLINE void store(half* ptr, f16x8 value, size_t n) {
  internal::partial_store_memcpy(ptr, value, n);
}
YNN_ALWAYS_INLINE void store(int16_t* ptr, s16x8 value, size_t n) {
  internal::partial_store_memcpy(ptr, value, n);
}
YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x16 value, size_t n) {
  internal::partial_store_memcpy(ptr, value, n);
}
YNN_ALWAYS_INLINE void store(int8_t* ptr, s8x16 value, size_t n) {
  internal::partial_store_memcpy(ptr, value, n);
}

YNN_ALWAYS_INLINE void store(half* ptr, f16x4 value, size_t n) {
  internal::partial_store_memcpy(ptr, value, n);
}
YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x8 value, size_t n) {
  internal::partial_store_memcpy(ptr, value, n);
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_ARM_NEON_PARTIAL_LOAD_STORE_H_
