// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_ARM_H_
#define XNNPACK_YNNPACK_BASE_SIMD_ARM_H_

#include <arm_neon.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/multi_vec.h"
#include "ynnpack/base/simd/vec.h"

namespace ynn {

namespace simd {

template <>
struct vec<float, 4> {
  using value_type = float;
  static constexpr std::integral_constant<size_t, 4> N = {};

  vec() = default;
  explicit vec(float32x4_t v) : v(v) {}
  vec(float x) : v(vdupq_n_f32(x)) {}  // NOLINT

  float32x4_t v;
};

template <>
struct vec<int32_t, 4> {
  using value_type = int32_t;
  static constexpr std::integral_constant<size_t, 4> N = {};

  vec() = default;
  explicit vec(int32x4_t v) : v(v) {}
  vec(int32_t x) : v(vdupq_n_s32(x)) {}  // NOLINT

  int32x4_t v;
};

template <>
struct vec<bfloat16, 8> {
  using value_type = bfloat16;
  static constexpr std::integral_constant<size_t, 8> N = {};

  vec() = default;
  explicit vec(uint16x8_t v) : v(v) {}
  vec(bfloat16 x) : v(vdupq_n_u16(x.to_bits())) {}  // NOLINT

  uint16x8_t v;
};

template <>
struct vec<half, 8> {
  using value_type = half;
  static constexpr std::integral_constant<size_t, 8> N = {};

  vec() = default;
  explicit vec(uint16x8_t v) : v(v) {}
  vec(half x) : v(vdupq_n_u16(x.to_bits())) {}  // NOLINT

  uint16x8_t v;
};

template <>
struct vec<int16_t, 8> {
  using value_type = int16_t;
  static constexpr std::integral_constant<size_t, 8> N = {};

  vec() = default;
  explicit vec(int16x8_t v) : v(v) {}
  vec(int16_t x) : v(vdupq_n_s16(x)) {}  // NOLINT

  int16x8_t v;
};

template <>
struct vec<uint8_t, 16> {
  using value_type = uint8_t;
  static constexpr std::integral_constant<size_t, 16> N = {};

  vec() = default;
  explicit vec(uint8x16_t v) : v(v) {}
  vec(uint8_t x) : v(vdupq_n_u8(x)) {}  // NOLINT

  uint8x16_t v;
};

template <>
struct vec<int8_t, 16> {
  using value_type = int8_t;
  static constexpr std::integral_constant<size_t, 16> N = {};

  vec() = default;
  explicit vec(int8x16_t v) : v(v) {}
  vec(int8_t x) : v(vdupq_n_s8(x)) {}  // NOLINT

  int8x16_t v;
};

using f32x4 = vec<float, 4>;
using s32x4 = vec<int32_t, 4>;
using bf16x8 = vec<bfloat16, 8>;
using f16x8 = vec<half, 8>;
using s16x8 = vec<int16_t, 8>;
using u8x16 = vec<uint8_t, 16>;
using s8x16 = vec<int8_t, 16>;

namespace internal {

YNN_ALWAYS_INLINE int32x4x2_t vtrn(int32x4_t a, int32x4_t b) {
  return vtrnq_s32(a, b);
}
YNN_ALWAYS_INLINE uint32x4x2_t vtrn(uint32x4_t a, uint32x4_t b) {
  return vtrnq_u32(a, b);
}
YNN_ALWAYS_INLINE float32x4x2_t vtrn(float32x4_t a, float32x4_t b) {
  return vtrnq_f32(a, b);
}

YNN_ALWAYS_INLINE int32x4_t vcombine(int32x2_t a, int32x2_t b) {
  return vcombine_s32(a, b);
}
YNN_ALWAYS_INLINE uint32x4_t vcombine(uint32x2_t a, uint32x2_t b) {
  return vcombine_u32(a, b);
}
YNN_ALWAYS_INLINE float32x4_t vcombine(float32x2_t a, float32x2_t b) {
  return vcombine_f32(a, b);
}

YNN_ALWAYS_INLINE int32x2_t vget_low(int32x4_t a) { return vget_low_s32(a); }
YNN_ALWAYS_INLINE int32x2_t vget_high(int32x4_t a) { return vget_high_s32(a); }
YNN_ALWAYS_INLINE uint32x2_t vget_low(uint32x4_t a) { return vget_low_u32(a); }
YNN_ALWAYS_INLINE uint32x2_t vget_high(uint32x4_t a) {
  return vget_high_u32(a);
}
YNN_ALWAYS_INLINE float32x2_t vget_low(float32x4_t a) {
  return vget_low_f32(a);
}
YNN_ALWAYS_INLINE float32x2_t vget_high(float32x4_t a) {
  return vget_high_f32(a);
}

template <int Lane>
YNN_ALWAYS_INLINE void vst1_lane(int32_t* ptr, int32x2_t v) {
  vst1_lane_s32(ptr, v, Lane);
}
template <int Lane>
YNN_ALWAYS_INLINE void vst1_lane(float* ptr, float32x2_t v) {
  vst1_lane_f32(ptr, v, Lane);
}

}  // namespace internal

YNN_ALWAYS_INLINE f32x4 load_aligned(const float* ptr, f32x4,
                                     decltype(f32x4::N) = {}) {
  return f32x4{vld1q_f32(ptr)};
}
YNN_ALWAYS_INLINE s32x4 load_aligned(const int32_t* ptr, s32x4,
                                     decltype(s32x4::N) = {}) {
  return s32x4{vld1q_s32(ptr)};
}
YNN_ALWAYS_INLINE bf16x8 load_aligned(const bfloat16* ptr, bf16x8,
                                      decltype(bf16x8::N) = {}) {
  return bf16x8{vld1q_u16(reinterpret_cast<const uint16_t*>(ptr))};
}
YNN_ALWAYS_INLINE f16x8 load_aligned(const half* ptr, f16x8,
                                     decltype(f16x8::N) = {}) {
  return f16x8{vld1q_u16(reinterpret_cast<const uint16_t*>(ptr))};
}
YNN_ALWAYS_INLINE s16x8 load_aligned(const int16_t* ptr, s16x8,
                                     decltype(s16x8::N) = {}) {
  return s16x8{vld1q_s16(ptr)};
}
YNN_ALWAYS_INLINE u8x16 load_aligned(const uint8_t* ptr, u8x16,
                                     decltype(u8x16::N) = {}) {
  return u8x16{vld1q_u8(ptr)};
}
YNN_ALWAYS_INLINE s8x16 load_aligned(const int8_t* ptr, s8x16,
                                     decltype(s8x16::N) = {}) {
  return s8x16{vld1q_s8(ptr)};
}

YNN_ALWAYS_INLINE void store_aligned(float* ptr, f32x4 b,
                                     decltype(f32x4::N) = {}) {
  vst1q_f32(ptr, b.v);
}
YNN_ALWAYS_INLINE void store_aligned(int32_t* ptr, s32x4 b,
                                     decltype(s32x4::N) = {}) {
  vst1q_s32(ptr, b.v);
}
YNN_ALWAYS_INLINE void store_aligned(bfloat16* ptr, bf16x8 b,
                                     decltype(bf16x8::N) = {}) {
  vst1q_u16(reinterpret_cast<uint16_t*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store_aligned(half* ptr, f16x8 b,
                                     decltype(f16x8::N) = {}) {
  vst1q_u16(reinterpret_cast<uint16_t*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store_aligned(int16_t* ptr, s16x8 b,
                                     decltype(s16x8::N) = {}) {
  vst1q_s16(ptr, b.v);
}
YNN_ALWAYS_INLINE void store_aligned(uint8_t* ptr, u8x16 b,
                                     decltype(u8x16::N) = {}) {
  vst1q_u8(ptr, b.v);
}
YNN_ALWAYS_INLINE void store_aligned(int8_t* ptr, s8x16 b,
                                     decltype(s8x16::N) = {}) {
  vst1q_s8(ptr, b.v);
}

YNN_ALWAYS_INLINE f32x4 load(const float* ptr, f32x4, decltype(f32x4::N) = {}) {
  return f32x4{vld1q_f32(ptr)};
}
YNN_ALWAYS_INLINE s32x4 load(const int32_t* ptr, s32x4,
                             decltype(s32x4::N) = {}) {
  return s32x4{vld1q_s32(ptr)};
}
YNN_ALWAYS_INLINE bf16x8 load(const bfloat16* ptr, bf16x8,
                              decltype(f16x8::N) = {}) {
  return bf16x8{vld1q_u16(reinterpret_cast<const uint16_t*>(ptr))};
}
YNN_ALWAYS_INLINE f16x8 load(const half* ptr, f16x8, decltype(f16x8::N) = {}) {
  return f16x8{vld1q_u16(reinterpret_cast<const uint16_t*>(ptr))};
}
YNN_ALWAYS_INLINE s16x8 load(const int16_t* ptr, s16x8,
                             decltype(s16x8::N) = {}) {
  return s16x8{vld1q_s16(ptr)};
}
YNN_ALWAYS_INLINE u8x16 load(const uint8_t* ptr, u8x16,
                             decltype(u8x16::N) = {}) {
  return u8x16{vld1q_u8(ptr)};
}
YNN_ALWAYS_INLINE s8x16 load(const int8_t* ptr, s8x16,
                             decltype(s8x16::N) = {}) {
  return s8x16{vld1q_s8(ptr)};
}

YNN_ALWAYS_INLINE void store(float* ptr, f32x4 b, decltype(f32x4::N) = {}) {
  vst1q_f32(ptr, b.v);
}
YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x4 b, decltype(s32x4::N) = {}) {
  vst1q_s32(ptr, b.v);
}
YNN_ALWAYS_INLINE void store(bfloat16* ptr, bf16x8 b,
                             decltype(bf16x8::N) = {}) {
  vst1q_u16(reinterpret_cast<uint16_t*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(half* ptr, f16x8 b, decltype(f16x8::N) = {}) {
  vst1q_u16(reinterpret_cast<uint16_t*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(int16_t* ptr, s16x8 b, decltype(s16x8::N) = {}) {
  vst1q_s16(ptr, b.v);
}
YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x16 b, decltype(u8x16::N) = {}) {
  vst1q_u8(ptr, b.v);
}
YNN_ALWAYS_INLINE void store(int8_t* ptr, s8x16 b, decltype(s8x16::N) = {}) {
  vst1q_s8(ptr, b.v);
}

namespace internal {

// Partial load/store with a non-constant number of elements.
template <typename T>
inline vec<T, 4> partial_load_lanes_x4(const T* ptr, vec<T, 4> src, size_t n) {
  assert(n < 4);
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
  return load_aligned(lanes, vec<T, 4>{});
}
template <typename T>
inline void partial_store_x32x4(T* ptr, vec<T, 4> b, size_t n) {
  assert(n < 4);
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

YNN_ALWAYS_INLINE f32x4 load(const float* ptr, f32x4 src, size_t n) {
  return internal::partial_load_lanes_x4(ptr, src, n);
}
YNN_ALWAYS_INLINE s32x4 load(const int32_t* ptr, s32x4 src, size_t n) {
  return internal::partial_load_lanes_x4(ptr, src, n);
}
YNN_ALWAYS_INLINE void store(float* ptr, f32x4 b, size_t n) {
  internal::partial_store_x32x4(ptr, b, n);
}
YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x4 b, size_t n) {
  internal::partial_store_x32x4(ptr, b, n);
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
YNN_ALWAYS_INLINE void store(bfloat16* ptr, bf16x8 value, size_t n) {
  internal::partial_store_memcpy(ptr, value, n);
}
YNN_ALWAYS_INLINE void store(half* ptr, f16x8 value, size_t n) {
  internal::partial_store_memcpy(ptr, value, n);
}
YNN_ALWAYS_INLINE void store(int16_t* ptr, s16x8 value, size_t n) {
  internal::partial_store_memcpy(ptr, value, n);
}

YNN_ALWAYS_INLINE u8x16 load(const uint8_t* ptr, u8x16 src, size_t n) {
  return internal::partial_load_memcpy(ptr, src, n);
}
YNN_ALWAYS_INLINE s8x16 load(const int8_t* ptr, s8x16 src, size_t n) {
  return internal::partial_load_memcpy(ptr, src, n);
}

YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x16 value, size_t n) {
  internal::partial_store_memcpy(ptr, value, n);
}
YNN_ALWAYS_INLINE void store(int8_t* ptr, s8x16 value, size_t n) {
  internal::partial_store_memcpy(ptr, value, n);
}

YNN_ALWAYS_INLINE f32x4& operator+=(f32x4& a, f32x4 b) {
  a.v = vaddq_f32(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s32x4& operator+=(s32x4& a, s32x4 b) {
  a.v = vaddq_s32(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s8x16& operator+=(s8x16& a, s8x16 b) {
  a.v = vaddq_s8(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE u8x16& operator+=(u8x16& a, u8x16 b) {
  a.v = vaddq_u8(a.v, b.v);
  return a;
}

YNN_ALWAYS_INLINE f32x4& operator-=(f32x4& a, f32x4 b) {
  a.v = vsubq_f32(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s32x4& operator-=(s32x4& a, s32x4 b) {
  a.v = vsubq_s32(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s8x16& operator-=(s8x16& a, s8x16 b) {
  a.v = vsubq_s8(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE u8x16& operator-=(u8x16& a, u8x16 b) {
  a.v = vsubq_u8(a.v, b.v);
  return a;
}

YNN_ALWAYS_INLINE f32x4& operator*=(f32x4& a, f32x4 b) {
  a.v = vmulq_f32(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s32x4& operator*=(s32x4& a, s32x4 b) {
  a.v = vmulq_s32(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s8x16& operator*=(s8x16& a, s8x16 b) {
  a.v = vmulq_s8(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE u8x16& operator*=(u8x16& a, u8x16 b) {
  a.v = vmulq_u8(a.v, b.v);
  return a;
}

YNN_ALWAYS_INLINE f32x4 operator+(f32x4 a, f32x4 b) { return a += b; }
YNN_ALWAYS_INLINE s32x4 operator+(s32x4 a, s32x4 b) { return a += b; }
YNN_ALWAYS_INLINE u8x16 operator+(u8x16 a, u8x16 b) { return a += b; }
YNN_ALWAYS_INLINE s8x16 operator+(s8x16 a, s8x16 b) { return a += b; }

YNN_ALWAYS_INLINE f32x4 operator-(f32x4 a, f32x4 b) { return a -= b; }
YNN_ALWAYS_INLINE s32x4 operator-(s32x4 a, s32x4 b) { return a -= b; }
YNN_ALWAYS_INLINE u8x16 operator-(u8x16 a, u8x16 b) { return a -= b; }
YNN_ALWAYS_INLINE s8x16 operator-(s8x16 a, s8x16 b) { return a -= b; }

YNN_ALWAYS_INLINE s16x8 operator&(s16x8 a, int b) {
  return s16x8{vandq_s16(a.v, vdupq_n_s16(b))};
}
YNN_ALWAYS_INLINE s16x8 operator>>(s16x8 a, int b) {
  return s16x8{vshlq_s16(a.v, vdupq_n_s16(-b))};
}
YNN_ALWAYS_INLINE s16x8 operator^(s16x8 a, s16x8 b) {
  return s16x8{veorq_s16(a.v, b.v)};
}

YNN_ALWAYS_INLINE f32x4 operator*(f32x4 a, f32x4 b) { return a *= b; }
YNN_ALWAYS_INLINE s32x4 operator*(s32x4 a, s32x4 b) { return a *= b; }
YNN_ALWAYS_INLINE u8x16 operator*(u8x16 a, u8x16 b) { return a *= b; }
YNN_ALWAYS_INLINE s8x16 operator*(s8x16 a, s8x16 b) { return a *= b; }

YNN_ALWAYS_INLINE f32x4 min(f32x4 a, f32x4 b) {
  return f32x4{vminq_f32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x4 min(s32x4 a, s32x4 b) {
  return s32x4{vminq_s32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 min(s16x8 a, s16x8 b) {
  return s16x8{vminq_s16(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 min(u8x16 a, u8x16 b) {
  return u8x16{vminq_u8(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 min(s8x16 a, s8x16 b) {
  return s8x16{vminq_s8(a.v, b.v)};
}

YNN_ALWAYS_INLINE f32x4 max(f32x4 a, f32x4 b) {
  return f32x4{vmaxq_f32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x4 max(s32x4 a, s32x4 b) {
  return s32x4{vmaxq_s32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 max(s16x8 a, s16x8 b) {
  return s16x8{vmaxq_s16(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 max(u8x16 a, u8x16 b) {
  return u8x16{vmaxq_u8(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 max(s8x16 a, s8x16 b) {
  return s8x16{vmaxq_s8(a.v, b.v)};
}

#ifdef YNN_ARCH_ARM32
YNN_ALWAYS_INLINE float vmaxvq_f32(float32x4_t a) {
  float32x2_t max_halves = vmax_f32(vget_low_f32(a), vget_high_f32(a));
  return vget_lane_f32(vpmax_f32(max_halves, max_halves), 0);
}
YNN_ALWAYS_INLINE float vminvq_f32(float32x4_t a) {
  float32x2_t min_halves = vmin_f32(vget_low_f32(a), vget_high_f32(a));
  return vget_lane_f32(vpmin_f32(min_halves, min_halves), 0);
}

YNN_ALWAYS_INLINE int16_t vmaxvq_s16(int16x8_t a) {
  int16x4_t max_halves = vmax_s16(vget_low_s16(a), vget_high_s16(a));
  int16x4_t max_pairs = vpmax_s16(max_halves, max_halves);
  return vget_lane_s16(vpmax_s16(max_pairs, max_pairs), 0);
}
YNN_ALWAYS_INLINE int16_t vminvq_s16(int16x8_t a) {
  int16x4_t min_halves = vmin_s16(vget_low_s16(a), vget_high_s16(a));
  int16x4_t min_pairs = vpmin_s16(min_halves, min_halves);
  return vget_lane_s16(vpmin_s16(min_pairs, min_pairs), 0);
}

YNN_ALWAYS_INLINE int8_t vmaxvq_s8(int8x16_t a) {
  int8x8_t max_halves = vmax_s8(vget_low_s8(a), vget_high_s8(a));
  int8x8_t max_pairs = vpmax_s8(max_halves, max_halves);
  int8x8_t max_quads = vpmax_s8(max_pairs, max_pairs);
  return vget_lane_s8(vpmax_s8(max_quads, max_quads), 0);
}
YNN_ALWAYS_INLINE int8_t vminvq_s8(int8x16_t a) {
  int8x8_t min_halves = vmin_s8(vget_low_s8(a), vget_high_s8(a));
  int8x8_t min_pairs = vpmin_s8(min_halves, min_halves);
  int8x8_t min_quads = vpmin_s8(min_pairs, min_pairs);
  return vget_lane_s8(vpmin_s8(min_quads, min_quads), 0);
}

YNN_ALWAYS_INLINE uint8_t vmaxvq_u8(uint8x16_t a) {
  uint8x8_t max_halves = vmax_u8(vget_low_u8(a), vget_high_u8(a));
  uint8x8_t max_pairs = vpmax_u8(max_halves, max_halves);
  uint8x8_t max_quads = vpmax_u8(max_pairs, max_pairs);
  return vget_lane_u8(vpmax_u8(max_quads, max_quads), 0);
}
YNN_ALWAYS_INLINE uint8_t vminvq_u8(uint8x16_t a) {
  uint8x8_t min_halves = vmin_u8(vget_low_u8(a), vget_high_u8(a));
  uint8x8_t min_pairs = vpmin_u8(min_halves, min_halves);
  uint8x8_t min_quads = vpmin_u8(min_pairs, min_pairs);
  return vget_lane_u8(vpmin_u8(min_quads, min_quads), 0);
}
#endif

YNN_ALWAYS_INLINE int8_t horizontal_max(s8x16 a) { return vmaxvq_s8(a.v); }
YNN_ALWAYS_INLINE uint8_t horizontal_max(u8x16 a) { return vmaxvq_u8(a.v); }
YNN_ALWAYS_INLINE int16_t horizontal_max(s16x8 a) { return vmaxvq_s16(a.v); }
YNN_ALWAYS_INLINE int32_t horizontal_max(s32x4 a) {
#ifndef __aarch64__
  int32x2_t lohi = vmax_s32(vget_low_s32(a.v), vget_high_s32(a.v));
  return std::max(vget_lane_s32(lohi, 0), vget_lane_s32(lohi, 1));
#else
  return vmaxvq_s32(a.v);
#endif
}
YNN_ALWAYS_INLINE float horizontal_max(f32x4 a) { return vmaxvq_f32(a.v); }
YNN_ALWAYS_INLINE int8_t horizontal_min(s8x16 a) { return vminvq_s8(a.v); }
YNN_ALWAYS_INLINE uint8_t horizontal_min(u8x16 a) { return vminvq_u8(a.v); }
YNN_ALWAYS_INLINE int16_t horizontal_min(s16x8 a) { return vminvq_s16(a.v); }
YNN_ALWAYS_INLINE int32_t horizontal_min(s32x4 a) {
#ifndef __aarch64__
  int32x2_t lohi = vmin_s32(vget_low_s32(a.v), vget_high_s32(a.v));
  return std::min(vget_lane_s32(lohi, 0), vget_lane_s32(lohi, 1));
#else
  return vminvq_s32(a.v);
#endif
}
YNN_ALWAYS_INLINE float horizontal_min(f32x4 a) { return vminvq_f32(a.v); }

template <typename T>
YNN_ALWAYS_INLINE std::array<vec<T, 4>, 4> transpose(
    std::array<vec<T, 4>, 4> x) {
  using internal::vcombine;
  using internal::vget_high;
  using internal::vget_low;
  using internal::vtrn;

  auto t01 = vtrn(x[0].v, x[1].v);
  auto t23 = vtrn(x[2].v, x[3].v);
  return {{
      vec<T, 4>{vcombine(vget_low(t01.val[0]), vget_low(t23.val[0]))},
      vec<T, 4>{vcombine(vget_low(t01.val[1]), vget_low(t23.val[1]))},
      vec<T, 4>{vcombine(vget_high(t01.val[0]), vget_high(t23.val[0]))},
      vec<T, 4>{vcombine(vget_high(t01.val[1]), vget_high(t23.val[1]))},
  }};
}

using f32x8 = multi_vec<f32x4, 2>;
using s32x8 = multi_vec<s32x4, 2>;
using s16x16 = multi_vec<s16x8, 2>;
using s32x16 = multi_vec<s32x4, 4>;

YNN_ALWAYS_INLINE f32x8 convert(bf16x8 a, float) {
  uint16x8x2_t a_u32 = vzipq_u16(vdupq_n_u16(0), a.v);
  return {
      f32x4{vreinterpretq_f32_u32(vreinterpretq_u32_u16(a_u32.val[0]))},
      f32x4{vreinterpretq_f32_u32(vreinterpretq_u32_u16(a_u32.val[1]))},
  };
}

YNN_ALWAYS_INLINE s16x16 convert(s8x16 b, int16_t) {
  return {
      s16x8{vmovl_s8(vget_low_s8(b.v))},
      s16x8{vmovl_s8(vget_high_s8(b.v))},
  };
}

YNN_ALWAYS_INLINE s16x16 convert(u8x16 b, int16_t) {
  return {
      s16x8{vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b.v)))},
      s16x8{vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b.v)))},
  };
}

YNN_ALWAYS_INLINE s32x8 convert(s16x8 b, int32_t) {
  return {
      s32x4{vmovl_s16(vget_low_s16(b.v))},
      s32x4{vmovl_s16(vget_high_s16(b.v))},
  };
}

YNN_ALWAYS_INLINE s32x16 convert(s8x16 b, int32_t) {
  s16x16 b_s16 = convert(b, int16_t{});
  s32x8 lo = convert(extract<0>(b_s16, s16x8{}), int32_t{});
  s32x8 hi = convert(extract<1>(b_s16, s16x8{}), int32_t{});
  return {
      extract<0>(lo, s32x4{}),
      extract<1>(lo, s32x4{}),
      extract<0>(hi, s32x4{}),
      extract<1>(hi, s32x4{}),
  };
}

YNN_ALWAYS_INLINE s32x16 convert(u8x16 b, int32_t) {
  s16x16 b_s16 = convert(b, int16_t{});
  s32x8 lo = convert(extract<0>(b_s16, s16x8{}), int32_t{});
  s32x8 hi = convert(extract<1>(b_s16, s16x8{}), int32_t{});
  return {
      extract<0>(lo, s32x4{}),
      extract<1>(lo, s32x4{}),
      extract<0>(hi, s32x4{}),
      extract<1>(hi, s32x4{}),
  };
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_ARM_H_
