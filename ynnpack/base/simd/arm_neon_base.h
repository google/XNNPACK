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
#include <tuple>
#include <type_traits>

#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/vec.h"

namespace ynn {

namespace simd {

// Half-vector wrappers
template <>
struct vec<uint8_t, 8> {
  using value_type = uint8_t;
  static constexpr std::integral_constant<size_t, 8> N = {};

  vec() = default;
  explicit vec(uint8x8_t v) : v(v) {}
  vec(uint8_t x) : v(vdup_n_u8(x)) {}  // NOLINT

  uint8x8_t v;
};

template <>
struct vec<float, 2> {
  using value_type = float;
  static constexpr std::integral_constant<size_t, 2> N = {};

  vec() = default;
  explicit vec(float32x2_t v) : v(v) {}
  vec(float x) : v(vdup_n_f32(x)) {}  // NOLINT

  float32x2_t v;
};

template <>
struct vec<bfloat16, 4> {
  using value_type = bfloat16;
  static constexpr std::integral_constant<size_t, 4> N = {};

  vec() = default;
  explicit vec(uint16x4_t v) : v(v) {}
  vec(bfloat16 x) : v(vdup_n_u16(x.to_bits())) {}  // NOLINT

  uint16x4_t v;
};

template <>
struct vec<half, 4> {
  using value_type = half;
  static constexpr std::integral_constant<size_t, 4> N = {};

  vec() = default;
  explicit vec(uint16x4_t v) : v(v) {}
  vec(half x) : v(vdup_n_u16(x.to_bits())) {}  // NOLINT

  uint16x4_t v;
};

using u8x8 = vec<uint8_t, 8>;
using f32x2 = vec<float, 2>;
using bf16x4 = vec<bfloat16, 4>;
using f16x4 = vec<half, 4>;

// Full vector wrappers
template <>
struct vec<float, 4> {
  using value_type = float;
  static constexpr std::integral_constant<size_t, 4> N = {};

  vec() = default;
  explicit vec(float32x4_t v) : v(v) {}
  vec(float x) : v(vdupq_n_f32(x)) {}  // NOLINT
  vec(f32x2 lo, f32x2 hi) : v(vcombine_f32(lo.v, hi.v)) {}

  float32x4_t v;
};

#ifdef YNN_ARCH_ARM64
template <>
struct vec<double, 2> {
  using value_type = double;
  static constexpr std::integral_constant<size_t, 2> N = {};

  vec() = default;
  explicit vec(float64x2_t v) : v(v) {}
  vec(double x) : v(vdupq_n_f64(x)) {}  // NOLINT
  vec(vec<double, 1> lo, vec<double, 1> hi) : v{lo.v, hi.v} {}

  float64x2_t v;
};

template <>
struct vec<int64_t, 2> {
  using value_type = int64_t;
  static constexpr std::integral_constant<size_t, 2> N = {};

  vec() = default;
  explicit vec(int64x2_t v) : v(v) {}
  vec(int64_t x) : v(vdupq_n_s64(x)) {}  // NOLINT
  vec(vec<int64_t, 1> lo, vec<int64_t, 1> hi) : v{lo.v, hi.v} {}

  int64x2_t v;
};
#endif

template <>
struct vec<uint32_t, 4> {
  using value_type = uint32_t;
  static constexpr std::integral_constant<size_t, 4> N = {};

  vec() = default;
  explicit vec(uint32x4_t v) : v(v) {}
  vec(uint32_t x) : v(vdupq_n_u32(x)) {}  // NOLINT

  uint32x4_t v;
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
  vec(bf16x4 lo, bf16x4 hi) : v(vcombine_u16(lo.v, hi.v)) {}

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
struct vec<uint16_t, 8> {
  using value_type = uint16_t;
  static constexpr std::integral_constant<size_t, 8> N = {};

  vec() = default;
  explicit vec(uint16x8_t v) : v(v) {}
  vec(uint16_t x) : v(vdupq_n_u16(x)) {}  // NOLINT

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
  vec(u8x8 lo, u8x8 hi) : v(vcombine_u8(lo.v, hi.v)) {}
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
#ifdef YNN_ARCH_ARM64
using f64x2 = vec<double, 2>;
using s64x2 = vec<int64_t, 2>;
#endif
using u32x4 = vec<uint32_t, 4>;
using s32x4 = vec<int32_t, 4>;
using bf16x8 = vec<bfloat16, 8>;
using f16x8 = vec<half, 8>;
using u16x8 = vec<uint16_t, 8>;
using s16x8 = vec<int16_t, 8>;
using u8x16 = vec<uint8_t, 16>;
using s8x16 = vec<int8_t, 16>;

YNN_ALWAYS_INLINE f32x2 lo(f32x4 x) { return f32x2{vget_low_f32(x.v)}; }
YNN_ALWAYS_INLINE f32x2 hi(f32x4 x) { return f32x2{vget_high_f32(x.v)}; }
YNN_ALWAYS_INLINE bf16x4 lo(bf16x8 x) { return bf16x4{vget_low_u16(x.v)}; }
YNN_ALWAYS_INLINE bf16x4 hi(bf16x8 x) { return bf16x4{vget_high_u16(x.v)}; }
YNN_ALWAYS_INLINE u8x8 lo(u8x16 x) { return u8x8{vget_low_u8(x.v)}; }
YNN_ALWAYS_INLINE u8x8 hi(u8x16 x) { return u8x8{vget_high_u8(x.v)}; }
#ifdef YNN_ARCH_ARM64
YNN_ALWAYS_INLINE vec<double, 1> lo(f64x2 x) {
  return vec<double, 1>{vgetq_lane_f64(x.v, 0)};
}
YNN_ALWAYS_INLINE vec<double, 1> hi(f64x2 x) {
  return vec<double, 1>{vgetq_lane_f64(x.v, 1)};
}
YNN_ALWAYS_INLINE vec<int64_t, 1> lo(s64x2 x) {
  return vec<int64_t, 1>{vgetq_lane_s64(x.v, 0)};
}
YNN_ALWAYS_INLINE vec<int64_t, 1> hi(s64x2 x) {
  return vec<int64_t, 1>{vgetq_lane_s64(x.v, 1)};
}
#endif

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

#ifdef YNN_ARCH_ARM64
template <int Lane>
YNN_ALWAYS_INLINE void vst1q_lane(double* ptr, float64x2_t v) {
  vst1q_lane_f64(ptr, v, Lane);
}

template <int Lane>
YNN_ALWAYS_INLINE float64x2_t vld1q_lane(const double* ptr, float64x2_t src) {
  return vld1q_lane_f64(ptr, src, Lane);
}
#endif

}  // namespace internal

YNN_ALWAYS_INLINE f32x4 load_aligned(const float* ptr, decltype(f32x4::N),
                                     f32x4 = {}) {
  return f32x4{vld1q_f32(ptr)};
}
YNN_ALWAYS_INLINE f32x2 load_aligned(const float* ptr, decltype(f32x2::N),
                                     f32x2 = {}) {
  return f32x2{vld1_f32(ptr)};
}
#ifdef YNN_ARCH_ARM64
YNN_ALWAYS_INLINE f64x2 load_aligned(const double* ptr, decltype(f64x2::N),
                                     f64x2 = {}) {
  return f64x2{vld1q_f64(ptr)};
}
#endif
YNN_ALWAYS_INLINE s32x4 load_aligned(const int32_t* ptr, decltype(s32x4::N),
                                     s32x4 = {}) {
  return s32x4{vld1q_s32(ptr)};
}
YNN_ALWAYS_INLINE bf16x8 load_aligned(const bfloat16* ptr, decltype(bf16x8::N),
                                      bf16x8 = {}) {
  return bf16x8{vld1q_u16(reinterpret_cast<const uint16_t*>(ptr))};
}
YNN_ALWAYS_INLINE f16x8 load_aligned(const half* ptr, decltype(f16x8::N),
                                     f16x8 = {}) {
  return f16x8{vld1q_u16(reinterpret_cast<const uint16_t*>(ptr))};
}
YNN_ALWAYS_INLINE s16x8 load_aligned(const int16_t* ptr, decltype(s16x8::N),
                                     s16x8 = {}) {
  return s16x8{vld1q_s16(ptr)};
}
YNN_ALWAYS_INLINE u8x16 load_aligned(const uint8_t* ptr, decltype(u8x16::N),
                                     u8x16 = {}) {
  return u8x16{vld1q_u8(ptr)};
}
YNN_ALWAYS_INLINE s8x16 load_aligned(const int8_t* ptr, decltype(s8x16::N),
                                     s8x16 = {}) {
  return s8x16{vld1q_s8(ptr)};
}

YNN_ALWAYS_INLINE bf16x4 load_aligned(const bfloat16* ptr, decltype(bf16x4::N),
                                      bf16x4 = {}) {
  return bf16x4{vld1_u16(reinterpret_cast<const uint16_t*>(ptr))};
}
YNN_ALWAYS_INLINE f16x4 load_aligned(const half* ptr, decltype(f16x4::N),
                                     f16x4 = {}) {
  return f16x4{vld1_u16(reinterpret_cast<const uint16_t*>(ptr))};
}
YNN_ALWAYS_INLINE u8x8 load_aligned(const uint8_t* ptr, decltype(u8x8::N),
                                    u8x8 = {}) {
  return u8x8{vld1_u8(ptr)};
}

YNN_ALWAYS_INLINE void store_aligned(float* ptr, f32x4 b,
                                     decltype(f32x4::N) = {}) {
  vst1q_f32(ptr, b.v);
}
YNN_ALWAYS_INLINE void store_aligned(float* ptr, f32x2 b,
                                     decltype(f32x2::N) = {}) {
  vst1_f32(ptr, b.v);
}
#ifdef YNN_ARCH_ARM64
YNN_ALWAYS_INLINE void store_aligned(double* ptr, f64x2 b,
                                     decltype(f64x2::N) = {}) {
  vst1q_f64(ptr, b.v);
}
#endif
YNN_ALWAYS_INLINE void store_aligned(uint32_t* ptr, u32x4 b,
                                     decltype(u32x4::N) = {}) {
  vst1q_u32(ptr, b.v);
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
YNN_ALWAYS_INLINE void store_aligned(uint16_t* ptr, u16x8 b,
                                     decltype(u16x8::N) = {}) {
  vst1q_u16(ptr, b.v);
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

YNN_ALWAYS_INLINE void store_aligned(bfloat16* ptr, bf16x4 b,
                                     decltype(bf16x4::N) = {}) {
  vst1_u16(reinterpret_cast<uint16_t*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store_aligned(half* ptr, f16x4 b,
                                     decltype(f16x4::N) = {}) {
  vst1_u16(reinterpret_cast<uint16_t*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store_aligned(uint8_t* ptr, u8x8 b,
                                     decltype(u8x8::N) = {}) {
  vst1_u8(ptr, b.v);
}

YNN_ALWAYS_INLINE f32x4 load(const float* ptr, decltype(f32x4::N), f32x4 = {}) {
  return f32x4{vld1q_f32(ptr)};
}
YNN_ALWAYS_INLINE f32x2 load(const float* ptr, decltype(f32x2::N), f32x2 = {}) {
  return f32x2{vld1_f32(ptr)};
}
#ifdef YNN_ARCH_ARM64
YNN_ALWAYS_INLINE f64x2 load(const double* ptr, decltype(f64x2::N),
                             f64x2 = {}) {
  return f64x2{vld1q_f64(ptr)};
}
#endif
YNN_ALWAYS_INLINE u32x4 load(const uint32_t* ptr, decltype(u32x4::N),
                             u32x4 = {}) {
  return u32x4{vld1q_u32(ptr)};
}
YNN_ALWAYS_INLINE s32x4 load(const int32_t* ptr, decltype(s32x4::N),
                             s32x4 = {}) {
  return s32x4{vld1q_s32(ptr)};
}
YNN_ALWAYS_INLINE bf16x8 load(const bfloat16* ptr, decltype(f16x8::N),
                              bf16x8 = {}) {
  return bf16x8{vld1q_u16(reinterpret_cast<const uint16_t*>(ptr))};
}
YNN_ALWAYS_INLINE f16x8 load(const half* ptr, decltype(f16x8::N), f16x8 = {}) {
  return f16x8{vld1q_u16(reinterpret_cast<const uint16_t*>(ptr))};
}
YNN_ALWAYS_INLINE u16x8 load(const uint16_t* ptr, decltype(u16x8::N),
                             u16x8 = {}) {
  return u16x8{vld1q_u16(ptr)};
}
YNN_ALWAYS_INLINE s16x8 load(const int16_t* ptr, decltype(s16x8::N),
                             s16x8 = {}) {
  return s16x8{vld1q_s16(ptr)};
}
YNN_ALWAYS_INLINE u8x16 load(const uint8_t* ptr, decltype(u8x16::N),
                             u8x16 = {}) {
  return u8x16{vld1q_u8(ptr)};
}
YNN_ALWAYS_INLINE s8x16 load(const int8_t* ptr, decltype(s8x16::N),
                             s8x16 = {}) {
  return s8x16{vld1q_s8(ptr)};
}

YNN_ALWAYS_INLINE bf16x4 load(const bfloat16* ptr, decltype(bf16x4::N),
                              bf16x4 = {}) {
  return bf16x4{vld1_u16(reinterpret_cast<const uint16_t*>(ptr))};
}
YNN_ALWAYS_INLINE f16x4 load(const half* ptr, decltype(f16x4::N), f16x4 = {}) {
  return f16x4{vld1_u16(reinterpret_cast<const uint16_t*>(ptr))};
}
YNN_ALWAYS_INLINE u8x8 load(const uint8_t* ptr, decltype(u8x8::N), u8x8 = {}) {
  return u8x8{vld1_u8(ptr)};
}

YNN_ALWAYS_INLINE void store(float* ptr, f32x4 b, decltype(f32x4::N) = {}) {
  vst1q_f32(ptr, b.v);
}
YNN_ALWAYS_INLINE void store(float* ptr, f32x2 b, decltype(f32x2::N) = {}) {
  vst1_f32(ptr, b.v);
}
#ifdef YNN_ARCH_ARM64
YNN_ALWAYS_INLINE void store(double* ptr, f64x2 b, decltype(f64x2::N) = {}) {
  vst1q_f64(ptr, b.v);
}
#endif
YNN_ALWAYS_INLINE void store(uint32_t* ptr, u32x4 b, decltype(u32x4::N) = {}) {
  vst1q_u32(ptr, b.v);
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
YNN_ALWAYS_INLINE void store(uint16_t* ptr, u16x8 b, decltype(u16x8::N) = {}) {
  vst1q_u16(ptr, b.v);
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

YNN_ALWAYS_INLINE void store(bfloat16* ptr, bf16x4 b,
                             decltype(bf16x4::N) = {}) {
  vst1_u16(reinterpret_cast<uint16_t*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(half* ptr, f16x4 b, decltype(f16x4::N) = {}) {
  vst1_u16(reinterpret_cast<uint16_t*>(ptr), b.v);
}
YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x8 b, decltype(u8x8::N) = {}) {
  vst1_u8(ptr, b.v);
}

YNN_ALWAYS_INLINE f32x4 operator+(f32x4 a, f32x4 b) {
  return f32x4{vaddq_f32(a.v, b.v)};
}
#ifdef YNN_ARCH_ARM64
YNN_ALWAYS_INLINE f64x2 operator+(f64x2 a, f64x2 b) {
  return f64x2{vaddq_f64(a.v, b.v)};
}
#endif
YNN_ALWAYS_INLINE s32x4 operator+(s32x4 a, s32x4 b) {
  return s32x4{vaddq_s32(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 operator+(u8x16 a, u8x16 b) {
  return u8x16{vaddq_u8(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 operator+(s8x16 a, s8x16 b) {
  return s8x16{vaddq_s8(a.v, b.v)};
}

YNN_ALWAYS_INLINE f32x4 operator-(f32x4 a, f32x4 b) {
  return f32x4{vsubq_f32(a.v, b.v)};
}
#ifdef YNN_ARCH_ARM64
YNN_ALWAYS_INLINE f64x2 operator-(f64x2 a, f64x2 b) {
  return f64x2{vsubq_f64(a.v, b.v)};
}
#endif
YNN_ALWAYS_INLINE s32x4 operator-(s32x4 a, s32x4 b) {
  return s32x4{vsubq_s32(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 operator-(u8x16 a, u8x16 b) {
  return u8x16{vsubq_u8(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 operator-(s8x16 a, s8x16 b) {
  return s8x16{vsubq_s8(a.v, b.v)};
}

YNN_ALWAYS_INLINE f32x4 operator*(f32x4 a, f32x4 b) {
  return f32x4{vmulq_f32(a.v, b.v)};
}
#ifdef YNN_ARCH_ARM64
YNN_ALWAYS_INLINE f64x2 operator*(f64x2 a, f64x2 b) {
  return f64x2{vmulq_f64(a.v, b.v)};
}
#endif

YNN_ALWAYS_INLINE f32x4 operator/(f32x4 a, f32x4 b) { return f32x4{a.v / b.v}; }
#ifdef YNN_ARCH_ARM64
YNN_ALWAYS_INLINE f64x2 operator/(f64x2 a, f64x2 b) {
  return f64x2{vdivq_f64(a.v, b.v)};
}
#endif

YNN_ALWAYS_INLINE f32x4 fma(f32x4 a, f32x4 b, f32x4 acc) {
#ifdef YNN_ARCH_ARM_NEONFMA
  return f32x4{vfmaq_f32(acc.v, a.v, b.v)};
#else
  return a * b + acc;
#endif
}
#ifdef YNN_ARCH_ARM64
YNN_ALWAYS_INLINE f64x2 fma(f64x2 a, f64x2 b, f64x2 acc) {
#ifdef YNN_ARCH_ARM_NEONFMA
  return f64x2{vfmaq_f64(acc.v, a.v, b.v)};
#else
  return a * b + acc;
#endif
}
#endif

YNN_ALWAYS_INLINE s32x4 operator*(s32x4 a, s32x4 b) {
  return s32x4{vmulq_s32(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 operator*(u8x16 a, u8x16 b) {
  return u8x16{vmulq_u8(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 operator*(s8x16 a, s8x16 b) {
  return s8x16{vmulq_s8(a.v, b.v)};
}

YNN_ALWAYS_INLINE s32x4 add_sat(s32x4 a, s32x4 b) {
  return s32x4{vqaddq_s32(a.v, b.v)};
}
YNN_ALWAYS_INLINE u32x4 add_sat(u32x4 a, u32x4 b) {
  return u32x4{vqaddq_u32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 add_sat(s16x8 a, s16x8 b) {
  return s16x8{vqaddq_s16(a.v, b.v)};
}
YNN_ALWAYS_INLINE u16x8 add_sat(u16x8 a, u16x8 b) {
  return u16x8{vqaddq_u16(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 add_sat(s8x16 a, s8x16 b) {
  return s8x16{vqaddq_s8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 add_sat(u8x16 a, u8x16 b) {
  return u8x16{vqaddq_u8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x8 add_sat(u8x8 a, u8x8 b) {
  return u8x8{vqadd_u8(a.v, b.v)};
}

YNN_ALWAYS_INLINE s32x4 sub_sat(s32x4 a, s32x4 b) {
  return s32x4{vqsubq_s32(a.v, b.v)};
}
YNN_ALWAYS_INLINE u32x4 sub_sat(u32x4 a, u32x4 b) {
  return u32x4{vqsubq_u32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 sub_sat(s16x8 a, s16x8 b) {
  return s16x8{vqsubq_s16(a.v, b.v)};
}
YNN_ALWAYS_INLINE u16x8 sub_sat(u16x8 a, u16x8 b) {
  return u16x8{vqsubq_u16(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 sub_sat(s8x16 a, s8x16 b) {
  return s8x16{vqsubq_s8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 sub_sat(u8x16 a, u8x16 b) {
  return u8x16{vqsubq_u8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x8 sub_sat(u8x8 a, u8x8 b) {
  return u8x8{vqsub_u8(a.v, b.v)};
}

YNN_ALWAYS_INLINE s16x8 operator>>(s16x8 a, int b) {
  return s16x8{vshlq_s16(a.v, vdupq_n_s16(-b))};
}
YNN_ALWAYS_INLINE s8x16 operator<<(s8x16 a, int b) {
  return s8x16{vshlq_s8(a.v, vdupq_n_s8(b))};
}
YNN_ALWAYS_INLINE s16x8 operator<<(s16x8 a, int b) {
  return s16x8{vshlq_s16(a.v, vdupq_n_s16(b))};
}
YNN_ALWAYS_INLINE s32x4 operator<<(s32x4 a, int b) {
  return s32x4{vshlq_s32(a.v, vdupq_n_s32(b))};
}

#ifdef YNN_ARCH_ARM64
YNN_ALWAYS_INLINE s64x2 operator&(s64x2 a, s64x2 b) {
  return s64x2{vandq_s64(a.v, b.v)};
}
YNN_ALWAYS_INLINE s64x2 operator|(s64x2 a, s64x2 b) {
  return s64x2{vorrq_s64(a.v, b.v)};
}
YNN_ALWAYS_INLINE s64x2 operator^(s64x2 a, s64x2 b) {
  return s64x2{veorq_s64(a.v, b.v)};
}
YNN_ALWAYS_INLINE s64x2 operator~(s64x2 a) {
  return s64x2{vreinterpretq_s64_s8(vmvnq_s8(vreinterpretq_s8_s64(a.v)))};
}
#endif  // YNN_ARCH_ARM64

YNN_ALWAYS_INLINE s32x4 operator&(s32x4 a, s32x4 b) {
  return s32x4{vandq_s32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x4 operator|(s32x4 a, s32x4 b) {
  return s32x4{vorrq_s32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x4 operator^(s32x4 a, s32x4 b) {
  return s32x4{veorq_s32(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x4 operator~(s32x4 a) { return s32x4{vmvnq_s32(a.v)}; }

YNN_ALWAYS_INLINE s16x8 operator&(s16x8 a, s16x8 b) {
  return s16x8{vandq_s16(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 operator|(s16x8 a, s16x8 b) {
  return s16x8{vorrq_s16(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 operator^(s16x8 a, s16x8 b) {
  return s16x8{veorq_s16(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 operator~(s16x8 a) { return s16x8{vmvnq_s16(a.v)}; }

YNN_ALWAYS_INLINE u8x16 operator&(u8x16 a, u8x16 b) {
  return u8x16{vandq_u8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 operator|(u8x16 a, u8x16 b) {
  return u8x16{vorrq_u8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 operator^(u8x16 a, u8x16 b) {
  return u8x16{veorq_u8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 operator~(u8x16 a) { return u8x16{vmvnq_u8(a.v)}; }

YNN_ALWAYS_INLINE s8x16 operator&(s8x16 a, s8x16 b) {
  return s8x16{vandq_s8(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 operator|(s8x16 a, s8x16 b) {
  return s8x16{vorrq_s8(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 operator^(s8x16 a, s8x16 b) {
  return s8x16{veorq_s8(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 operator~(s8x16 a) { return s8x16{vmvnq_s8(a.v)}; }

YNN_ALWAYS_INLINE u8x8 operator&(u8x8 a, u8x8 b) {
  return u8x8{vand_u8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x8 operator|(u8x8 a, u8x8 b) {
  return u8x8{vorr_u8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x8 operator^(u8x8 a, u8x8 b) {
  return u8x8{veor_u8(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x8 operator~(u8x8 a) { return u8x8{vmvn_u8(a.v)}; }

YNN_ALWAYS_INLINE f32x4 min(f32x4 a, f32x4 b) {
  return f32x4{vminq_f32(a.v, b.v)};
}
#ifdef YNN_ARCH_ARM64
YNN_ALWAYS_INLINE f64x2 min(f64x2 a, f64x2 b) {
  return f64x2{vminq_f64(a.v, b.v)};
}
#endif
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
#ifdef YNN_ARCH_ARM64
YNN_ALWAYS_INLINE f64x2 max(f64x2 a, f64x2 b) {
  return f64x2{vmaxq_f64(a.v, b.v)};
}
#endif
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

YNN_ALWAYS_INLINE f32x4 abs(f32x4 a) { return f32x4{vabsq_f32(a.v)}; }
#ifdef YNN_ARCH_ARM64
YNN_ALWAYS_INLINE f64x2 abs(f64x2 a) { return f64x2{vabsq_f64(a.v)}; }
#endif
YNN_ALWAYS_INLINE u32x4 abs(s32x4 a) {
  return u32x4{vreinterpretq_u32_s32(vabsq_s32(a.v))};
}
YNN_ALWAYS_INLINE u16x8 abs(s16x8 a) {
  return u16x8{vreinterpretq_u16_s16(vabsq_s16(a.v))};
}
YNN_ALWAYS_INLINE u8x16 abs(s8x16 a) {
  return u8x16{vreinterpretq_u8_s8(vabsq_s8(a.v))};
}

YNN_ALWAYS_INLINE s32x4 operator==(f32x4 a, f32x4 b) {
  return s32x4{vreinterpretq_s32_u32(vceqq_f32(a.v, b.v))};
}
YNN_ALWAYS_INLINE s32x4 operator!=(f32x4 a, f32x4 b) { return ~(a == b); }
YNN_ALWAYS_INLINE s32x4 operator<(f32x4 a, f32x4 b) {
  return s32x4{vreinterpretq_s32_u32(vcltq_f32(a.v, b.v))};
}
YNN_ALWAYS_INLINE s32x4 operator<=(f32x4 a, f32x4 b) {
  return s32x4{vreinterpretq_s32_u32(vcleq_f32(a.v, b.v))};
}
YNN_ALWAYS_INLINE s32x4 operator>(f32x4 a, f32x4 b) {
  return s32x4{vreinterpretq_s32_u32(vcgtq_f32(a.v, b.v))};
}
YNN_ALWAYS_INLINE s32x4 operator>=(f32x4 a, f32x4 b) {
  return s32x4{vreinterpretq_s32_u32(vcgeq_f32(a.v, b.v))};
}
YNN_ALWAYS_INLINE s32x4 isnan(f32x4 a) { return ~(a == a); }
YNN_ALWAYS_INLINE s32x4 isinf(f32x4 a) {
  uint32x4_t mask = vdupq_n_u32(0x7FFFFFFF);
  uint32x4_t inf = vdupq_n_u32(0x7F800000);
  return s32x4{vreinterpretq_s32_u32(
      vceqq_u32(vandq_u32(vreinterpretq_u32_f32(a.v), mask), inf))};
}
YNN_ALWAYS_INLINE s32x4 isfinite(f32x4 a) {
  uint32x4_t mask = vdupq_n_u32(0x7FFFFFFF);
  uint32x4_t inf = vdupq_n_u32(0x7F800000);
  return s32x4{vreinterpretq_s32_u32(
      vcltq_u32(vandq_u32(vreinterpretq_u32_f32(a.v), mask), inf))};
}

#ifdef YNN_ARCH_ARM64
YNN_ALWAYS_INLINE s64x2 operator==(f64x2 a, f64x2 b) {
  return s64x2{vreinterpretq_s64_u64(vceqq_f64(a.v, b.v))};
}
YNN_ALWAYS_INLINE s64x2 operator!=(f64x2 a, f64x2 b) { return ~(a == b); }
YNN_ALWAYS_INLINE s64x2 operator<(f64x2 a, f64x2 b) {
  return s64x2{vreinterpretq_s64_u64(vcltq_f64(a.v, b.v))};
}
YNN_ALWAYS_INLINE s64x2 operator<=(f64x2 a, f64x2 b) {
  return s64x2{vreinterpretq_s64_u64(vcleq_f64(a.v, b.v))};
}
YNN_ALWAYS_INLINE s64x2 operator>(f64x2 a, f64x2 b) {
  return s64x2{vreinterpretq_s64_u64(vcgtq_f64(a.v, b.v))};
}
YNN_ALWAYS_INLINE s64x2 operator>=(f64x2 a, f64x2 b) {
  return s64x2{vreinterpretq_s64_u64(vcgeq_f64(a.v, b.v))};
}
YNN_ALWAYS_INLINE s64x2 isnan(f64x2 a) { return ~(a == a); }
YNN_ALWAYS_INLINE s64x2 isinf(f64x2 a) {
  uint64x2_t mask = vdupq_n_u64(0x7FFFFFFFFFFFFFFFULL);
  uint64x2_t inf = vdupq_n_u64(0x7FF0000000000000ULL);
  return s64x2{vreinterpretq_s64_u64(
      vceqq_u64(vandq_u64(vreinterpretq_u64_f64(a.v), mask), inf))};
}
YNN_ALWAYS_INLINE s64x2 isfinite(f64x2 a) {
  uint64x2_t mask = vdupq_n_u64(0x7FFFFFFFFFFFFFFFULL);
  uint64x2_t inf = vdupq_n_u64(0x7FF0000000000000ULL);
  return s64x2{vreinterpretq_s64_u64(
      vcltq_u64(vandq_u64(vreinterpretq_u64_f64(a.v), mask), inf))};
}
#endif

YNN_ALWAYS_INLINE s32x4 operator==(s32x4 a, s32x4 b) {
  return s32x4{vreinterpretq_s32_u32(vceqq_s32(a.v, b.v))};
}
YNN_ALWAYS_INLINE s32x4 operator>(s32x4 a, s32x4 b) {
  return s32x4{vreinterpretq_s32_u32(vcgtq_s32(a.v, b.v))};
}
YNN_ALWAYS_INLINE s32x4 operator<(s32x4 a, s32x4 b) { return b > a; }

YNN_ALWAYS_INLINE s16x8 operator==(s16x8 a, s16x8 b) {
  return s16x8{vreinterpretq_s16_u16(vceqq_s16(a.v, b.v))};
}
YNN_ALWAYS_INLINE s16x8 operator>(s16x8 a, s16x8 b) {
  return s16x8{vreinterpretq_s16_u16(vcgtq_s16(a.v, b.v))};
}
YNN_ALWAYS_INLINE s16x8 operator<(s16x8 a, s16x8 b) { return b > a; }

YNN_ALWAYS_INLINE s8x16 operator==(s8x16 a, s8x16 b) {
  return s8x16{vreinterpretq_s8_u8(vceqq_s8(a.v, b.v))};
}
YNN_ALWAYS_INLINE s8x16 operator>(s8x16 a, s8x16 b) {
  return s8x16{vreinterpretq_s8_u8(vcgtq_s8(a.v, b.v))};
}
YNN_ALWAYS_INLINE s8x16 operator<(s8x16 a, s8x16 b) { return b > a; }

YNN_ALWAYS_INLINE f32x4 select(s32x4 cond, f32x4 a, f32x4 b) {
  return f32x4{vbslq_f32(vreinterpretq_u32_s32(cond.v), a.v, b.v)};
}
#ifdef YNN_ARCH_ARM64
YNN_ALWAYS_INLINE f64x2 select(s64x2 cond, f64x2 a, f64x2 b) {
  return f64x2{vbslq_f64(vreinterpretq_u64_s64(cond.v), a.v, b.v)};
}
#endif
YNN_ALWAYS_INLINE s32x4 select(s32x4 cond, s32x4 a, s32x4 b) {
  return s32x4{vreinterpretq_s32_u32(vbslq_u32(vreinterpretq_u32_s32(cond.v),
                                               vreinterpretq_u32_s32(a.v),
                                               vreinterpretq_u32_s32(b.v)))};
}
YNN_ALWAYS_INLINE u32x4 select(s32x4 cond, u32x4 a, u32x4 b) {
  return u32x4{vbslq_u32(vreinterpretq_u32_s32(cond.v), a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 select(s16x8 cond, s16x8 a, s16x8 b) {
  return s16x8{vreinterpretq_s16_u16(vbslq_u16(vreinterpretq_u16_s16(cond.v),
                                               vreinterpretq_u16_s16(a.v),
                                               vreinterpretq_u16_s16(b.v)))};
}
YNN_ALWAYS_INLINE u16x8 select(s16x8 cond, u16x8 a, u16x8 b) {
  return u16x8{vbslq_u16(vreinterpretq_u16_s16(cond.v), a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 select(s8x16 cond, s8x16 a, s8x16 b) {
  return s8x16{vreinterpretq_s8_u8(vbslq_u8(vreinterpretq_u8_s8(cond.v),
                                            vreinterpretq_u8_s8(a.v),
                                            vreinterpretq_u8_s8(b.v)))};
}
YNN_ALWAYS_INLINE u8x16 select(s8x16 cond, u8x16 a, u8x16 b) {
  return u8x16{vbslq_u8(vreinterpretq_u8_s8(cond.v), a.v, b.v)};
}

YNN_ALWAYS_INLINE f32x4 floor_log2(f32x4 a) {
  uint32x4_t is_zero = vceqq_f32(a.v, vdupq_n_f32(0.0f));
  a.v = vreinterpretq_f32_u32(
      vorrq_u32(vandq_u32(is_zero, vreinterpretq_u32_f32(vdupq_n_f32(-0.0f))),
                vreinterpretq_u32_f32(a.v)));

  uint32x4_t sign_and_exp_mask = vdupq_n_u32(0xFF800000);
  int32x4_t exp = vreinterpretq_s32_u32(
      vandq_u32(vreinterpretq_u32_f32(a.v), sign_and_exp_mask));

  float32x4_t infinity = vdupq_n_f32(std::numeric_limits<float>::infinity());
  uint32x4_t is_inf = vceqq_f32(a.v, infinity);

  exp = vshrq_n_s32(exp, 8);

  float32x4_t bias_256 = vdupq_n_f32(256.0f);
  float32x4_t bias_383 = vdupq_n_f32(383.0f);
  float32x4_t res =
      vsubq_f32(vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(bias_256),
                                                vreinterpretq_u32_s32(exp))),
                bias_383);
  return f32x4{vbslq_f32(is_inf, infinity, res)};
}
YNN_ALWAYS_INLINE f32x4 exp2_round(f32x4 a) {
#if defined(__ARM_ARCH) && __ARM_ARCH < 8
  const float result[] = {
      ynn::exp2_round(vgetq_lane_f32(a.v, 0)),
      ynn::exp2_round(vgetq_lane_f32(a.v, 1)),
      ynn::exp2_round(vgetq_lane_f32(a.v, 2)),
      ynn::exp2_round(vgetq_lane_f32(a.v, 3)),
  };
  return f32x4{vld1q_f32(result)};
#else
  float32x4_t magic = vdupq_n_f32(127.0f + static_cast<float>(1 << 23));
  int32x4_t res_bits = vreinterpretq_s32_f32(vaddq_f32(a.v, magic));
  return f32x4{vreinterpretq_f32_s32(vshlq_n_s32(res_bits, 23))};
#endif
}
#ifdef YNN_ARCH_ARM64
YNN_ALWAYS_INLINE f64x2 floor_log2(f64x2 a) {
  uint64x2_t is_zero = vceqq_f64(a.v, vdupq_n_f64(0.0));
  a.v = vreinterpretq_f64_u64(
      vorrq_u64(vandq_u64(is_zero, vreinterpretq_u64_f64(vdupq_n_f64(-0.0))),
                vreinterpretq_u64_f64(a.v)));

  uint64x2_t sign_and_exp_mask = vdupq_n_u64(0xFFF0000000000000ULL);
  int64x2_t exp = vreinterpretq_s64_u64(
      vandq_u64(vreinterpretq_u64_f64(a.v), sign_and_exp_mask));

  float64x2_t infinity = vdupq_n_f64(std::numeric_limits<double>::infinity());
  uint64x2_t is_inf = vceqq_f64(a.v, infinity);

  exp = vshrq_n_s64(exp, 11);

  float64x2_t bias_2048 = vdupq_n_f64(2048.0);
  float64x2_t bias_3071 = vdupq_n_f64(3071.0);
  float64x2_t res = vsubq_f64(
      vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(bias_2048),
                                      vreinterpretq_u64_s64(exp))),
      bias_3071);
  return f64x2{vbslq_f64(is_inf, infinity, res)};
}
YNN_ALWAYS_INLINE f64x2 exp2_round(f64x2 a) {
  float64x2_t magic = vdupq_n_f64(1023.0 + static_cast<double>(1ll << 52));
  int64x2_t res_bits = vreinterpretq_s64_f64(vaddq_f64(a.v, magic));
  return f64x2{vreinterpretq_f64_s64(vshlq_n_s64(res_bits, 52))};
}
#endif

namespace internal {
YNN_ALWAYS_INLINE float32x4_t and_f32(float32x4_t a, float32x4_t b) {
  return vreinterpretq_f32_u32(
      vandq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
}

YNN_ALWAYS_INLINE float32x4_t or_f32(float32x4_t a, float32x4_t b) {
  return vreinterpretq_f32_u32(
      vorrq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
}

YNN_ALWAYS_INLINE float32x4_t xor_f32(float32x4_t a, float32x4_t b) {
  return vreinterpretq_f32_u32(
      veorq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
}

YNN_ALWAYS_INLINE float32x4_t not_f32(float32x4_t a) {
  return vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(a)));
}
}  // namespace internal

YNN_ALWAYS_INLINE f32x4 floor(f32x4 a) {
#if defined(__ARM_ARCH) && __ARM_ARCH < 8
  float32x4_t max_non_int_val = vdupq_n_f32(static_cast<float>(1 << 23));
  uint32x4_t use_rounding = vcaltq_f32(a.v, max_non_int_val);
  float32x4_t trunc = vcvtq_f32_s32(vcvtq_s32_f32(a.v));
  uint32x4_t floor_mask = vcgtq_f32(trunc, a.v);
  float32x4_t one = vdupq_n_f32(1.0f);
  float32x4_t floored = vbslq_f32(floor_mask, vsubq_f32(trunc, one), trunc);
  return f32x4{vbslq_f32(use_rounding, floored, a.v)};
#else
  return f32x4{vrndmq_f32(a.v)};
#endif
}

#ifdef YNN_ARCH_ARM64
YNN_ALWAYS_INLINE f64x2 floor(f64x2 a) { return f64x2{vrndmq_f64(a.v)}; }
#endif

YNN_ALWAYS_INLINE f32x4 ceil(f32x4 a) {
#if defined(__ARM_ARCH) && __ARM_ARCH < 8
  float32x4_t max_non_int_val = vdupq_n_f32(static_cast<float>(1 << 23));
  uint32x4_t use_rounding = vcaltq_f32(a.v, max_non_int_val);
  float32x4_t trunc = vcvtq_f32_s32(vcvtq_s32_f32(a.v));
  uint32x4_t ceil_mask = vcltq_f32(trunc, a.v);
  float32x4_t one = vdupq_n_f32(1.0f);
  float32x4_t ceiled = vbslq_f32(ceil_mask, vaddq_f32(trunc, one), trunc);
  return f32x4{vbslq_f32(use_rounding, ceiled, a.v)};
#else
  return f32x4{vrndpq_f32(a.v)};
#endif
}

#ifdef YNN_ARCH_ARM64
YNN_ALWAYS_INLINE f64x2 ceil(f64x2 a) { return f64x2{vrndpq_f64(a.v)}; }
#endif

YNN_ALWAYS_INLINE f32x4 round(f32x4 a) {
#if defined(__ARM_ARCH) && __ARM_ARCH < 8
  float32x4_t max_non_int_val = vdupq_n_f32(static_cast<float>(1 << 23));
  float32x4_t filter = vreinterpretq_f32_u32(vcaltq_f32(a.v, max_non_int_val));
  float32x4_t half = vdupq_n_f32(0.5f);
  float32x4_t sign_mask = vdupq_n_f32(-0.0f);
  float32x4_t signed_half =
      internal::or_f32(internal::and_f32(a.v, sign_mask), half);
  float32x4_t result_away =
      vcvtq_f32_s32(vcvtq_s32_f32(vaddq_f32(a.v, signed_half)));
  // vresult_away is round ties away from zero.
  // We want round ties to even.
  // If vresult_away is odd, and it was a tie, we need to correct by 1 towards
  // 0.
  uint32x4_t tie = vceqq_f32(vabsq_f32(vsubq_f32(result_away, a.v)), half);
  int32x4_t i_result_away = vcvtq_s32_f32(result_away);
  uint32x4_t odd =
      vcgtq_s32(vandq_s32(i_result_away, vdupq_n_s32(1)), vdupq_n_s32(0));
  uint32x4_t correct_mask = vandq_u32(tie, odd);
  float32x4_t correction = internal::and_f32(
      vreinterpretq_f32_u32(correct_mask),
      internal::or_f32(internal::and_f32(a.v, sign_mask), vdupq_n_f32(1.0f)));
  float32x4_t result = vsubq_f32(result_away, correction);

  return f32x4{
      internal::or_f32(internal::and_f32(filter, result),
                       internal::and_f32(internal::not_f32(filter), a.v))};
#else
  return f32x4{vrndnq_f32(a.v)};
#endif
}

#ifdef YNN_ARCH_ARM64
YNN_ALWAYS_INLINE f64x2 round(f64x2 a) { return f64x2{vrndnq_f64(a.v)}; }
#endif

YNN_ALWAYS_INLINE f32x4 sqrt(f32x4 a) {
#ifndef YNN_ARCH_ARM64
  // Get the initial low-precision estimate of 1/sqrt(a).
  float32x4_t rsqrt_estimate = vrsqrteq_f32(a.v);

  // Perform one Newton-Raphson refinement
  rsqrt_estimate =
      vmulq_f32(vrsqrtsq_f32(vmulq_f32(a.v, rsqrt_estimate), rsqrt_estimate),
                rsqrt_estimate);
  // Perform second Newton-Raphson refinement
  rsqrt_estimate =
      vmulq_f32(vrsqrtsq_f32(vmulq_f32(a.v, rsqrt_estimate), rsqrt_estimate),
                rsqrt_estimate);

  float32x4_t sqrt_result = vmulq_f32(a.v, rsqrt_estimate);

  return f32x4{sqrt_result};
#else
  return f32x4{vsqrtq_f32(a.v)};
#endif
}

#ifdef YNN_ARCH_ARM64
YNN_ALWAYS_INLINE f64x2 sqrt(f64x2 a) { return f64x2{vsqrtq_f64(a.v)}; }
#endif

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
#ifndef YNN_ARCH_ARM64
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
#ifndef YNN_ARCH_ARM64
  int32x2_t lohi = vmin_s32(vget_low_s32(a.v), vget_high_s32(a.v));
  return std::min(vget_lane_s32(lohi, 0), vget_lane_s32(lohi, 1));
#else
  return vminvq_s32(a.v);
#endif
}
YNN_ALWAYS_INLINE float horizontal_min(f32x4 a) { return vminvq_f32(a.v); }

YNN_ALWAYS_INLINE float horizontal_sum(f32x4 a) {
  float32x2_t lohi = vadd_f32(vget_low_f32(a.v), vget_high_f32(a.v));
  return vget_lane_f32(lohi, 0) + vget_lane_f32(lohi, 1);
}
YNN_ALWAYS_INLINE int32_t horizontal_sum(s32x4 a) {
  int32x2_t lohi = vadd_s32(vget_low_s32(a.v), vget_high_s32(a.v));
  return vget_lane_s32(lohi, 0) + vget_lane_s32(lohi, 1);
}

#ifdef YNN_ARCH_ARM64
YNN_ALWAYS_INLINE double horizontal_min(f64x2 a) {
  return std::min(vgetq_lane_f64(a.v, 0), vgetq_lane_f64(a.v, 1));
}
YNN_ALWAYS_INLINE double horizontal_max(f64x2 a) {
  return std::max(vgetq_lane_f64(a.v, 0), vgetq_lane_f64(a.v, 1));
}
YNN_ALWAYS_INLINE double horizontal_sum(f64x2 a) {
  return vgetq_lane_f64(a.v, 0) + vgetq_lane_f64(a.v, 1);
}
#endif

YNN_ALWAYS_INLINE std::tuple<u8x16, u8x16> interleave(
    std::integral_constant<size_t, 64>, u8x16 x0, u8x16 x1) {
  return {
      u8x16{vcombine_u8(vget_low_u8(x0.v), vget_low_u8(x1.v))},
      u8x16{vcombine_u8(vget_high_u8(x0.v), vget_high_u8(x1.v))},
  };
}
YNN_ALWAYS_INLINE std::tuple<u8x16, u8x16> interleave(
    std::integral_constant<size_t, 32>, u8x16 x0, u8x16 x1) {
  uint32x4x2_t x01 =
      vzipq_u32(vreinterpretq_u32_u8(x0.v), vreinterpretq_u32_u8(x1.v));
  return {u8x16{vreinterpretq_u8_u32(x01.val[0])},
          u8x16{vreinterpretq_u8_u32(x01.val[1])}};
}
YNN_ALWAYS_INLINE std::tuple<u8x16, u8x16> interleave(
    std::integral_constant<size_t, 16>, u8x16 x0, u8x16 x1) {
  uint16x8x2_t x01 =
      vzipq_u16(vreinterpretq_u16_u8(x0.v), vreinterpretq_u16_u8(x1.v));
  return {u8x16{vreinterpretq_u8_u16(x01.val[0])},
          u8x16{vreinterpretq_u8_u16(x01.val[1])}};
}
YNN_ALWAYS_INLINE std::tuple<u8x16, u8x16> interleave(
    std::integral_constant<size_t, 8>, u8x16 x0, u8x16 x1) {
  uint8x16x2_t x01 = vzipq_u8(x0.v, x1.v);
  return {u8x16{x01.val[0]}, u8x16{x01.val[1]}};
}
YNN_ALWAYS_INLINE std::tuple<u8x16, u8x16> interleave(
    std::integral_constant<size_t, 4>, u8x16 x0, u8x16 x1) {
  return interleave(
      std::integral_constant<size_t, 8>{},
      u8x16{vbslq_u8(vdupq_n_u8(0xf0), vshlq_n_u8(x1.v, 4), x0.v)},
      u8x16{vbslq_u8(vdupq_n_u8(0xf0), x1.v, vshrq_n_u8(x0.v, 4))});
}
YNN_ALWAYS_INLINE std::tuple<u8x16, u8x16> interleave(
    std::integral_constant<size_t, 2>, u8x16 x0, u8x16 x1) {
  return interleave(
      std::integral_constant<size_t, 4>{},
      u8x16{vbslq_u8(vdupq_n_u8(0xcc), vshlq_n_u8(x1.v, 2), x0.v)},
      u8x16{vbslq_u8(vdupq_n_u8(0xcc), x1.v, vshrq_n_u8(x0.v, 2))});
}

YNN_ALWAYS_INLINE std::tuple<u8x8, u8x8> interleave(
    std::integral_constant<size_t, 32>, u8x8 x0, u8x8 x1) {
  uint32x2x2_t x01 =
      vzip_u32(vreinterpret_u32_u8(x0.v), vreinterpret_u32_u8(x1.v));
  return {u8x8{vreinterpret_u8_u32(x01.val[0])},
          u8x8{vreinterpret_u8_u32(x01.val[1])}};
}
YNN_ALWAYS_INLINE std::tuple<u8x8, u8x8> interleave(
    std::integral_constant<size_t, 16>, u8x8 x0, u8x8 x1) {
  uint16x4x2_t x01 =
      vzip_u16(vreinterpret_u16_u8(x0.v), vreinterpret_u16_u8(x1.v));
  return {u8x8{vreinterpret_u8_u16(x01.val[0])},
          u8x8{vreinterpret_u8_u16(x01.val[1])}};
}
YNN_ALWAYS_INLINE std::tuple<u8x8, u8x8> interleave(
    std::integral_constant<size_t, 8>, u8x8 x0, u8x8 x1) {
  uint8x8x2_t x01 = vzip_u8(x0.v, x1.v);
  return {u8x8{x01.val[0]}, u8x8{x01.val[1]}};
}
YNN_ALWAYS_INLINE std::tuple<u8x8, u8x8> interleave(
    std::integral_constant<size_t, 4>, u8x8 x0, u8x8 x1) {
  return interleave(std::integral_constant<size_t, 8>{},
                    u8x8{vbsl_u8(vdup_n_u8(0xf0), vshl_n_u8(x1.v, 4), x0.v)},
                    u8x8{vbsl_u8(vdup_n_u8(0xf0), x1.v, vshr_n_u8(x0.v, 4))});
}
YNN_ALWAYS_INLINE std::tuple<u8x8, u8x8> interleave(
    std::integral_constant<size_t, 2>, u8x8 x0, u8x8 x1) {
  return interleave(std::integral_constant<size_t, 4>{},
                    u8x8{vbsl_u8(vdup_n_u8(0xcc), vshl_n_u8(x1.v, 2), x0.v)},
                    u8x8{vbsl_u8(vdup_n_u8(0xcc), x1.v, vshr_n_u8(x0.v, 2))});
}

using f32x8 = vec<float, 8>;
#ifdef YNN_ARCH_ARM64
using f64x4 = vec<double, 4>;
using f64x8 = vec<double, 8>;
#endif
using s32x8 = vec<int32_t, 8>;
using s16x16 = vec<int16_t, 16>;
using s32x16 = vec<int32_t, 16>;
using f32x16 = vec<float, 16>;

YNN_ALWAYS_INLINE f32x4 cast(bf16x4 a, float) {
  return f32x4{vreinterpretq_f32_u32(vshll_n_u16(a.v, 16))};
}

YNN_ALWAYS_INLINE bf16x4 cast(f32x4 a, bfloat16) {
  uint32x4_t u = vreinterpretq_u32_f32(a.v);
  uint32x4_t is_nan = vcgtq_u32(vshlq_n_u32(u, 1), vdupq_n_u32(0xFF000000u));
#ifdef YNN_ARCH_ARM_NEONBF16
  uint16x4_t rounded = vreinterpret_u16_bf16(vcvt_bf16_f32(a.v));
#else
  uint32x4_t lsb = vandq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(1));
  uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7FFF), lsb);
  uint16x4_t rounded = vshrn_n_u32(vaddq_u32(u, bias), 16);
#endif
  uint16x4_t nan_res = vmovn_u32(vorrq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(1)));
  return bf16x4{vbsl_u16(vmovn_u32(is_nan), nan_res, rounded)};
}

YNN_ALWAYS_INLINE s16x16 cast(s8x16 b, int16_t) {
  return {
      s16x8{vmovl_s8(vget_low_s8(b.v))},
      s16x8{vmovl_s8(vget_high_s8(b.v))},
  };
}

YNN_ALWAYS_INLINE s16x16 cast(u8x16 b, int16_t) {
  return {
      s16x8{vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b.v)))},
      s16x8{vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b.v)))},
  };
}

YNN_ALWAYS_INLINE s32x8 cast(s16x8 b, int32_t) {
  return {
      s32x4{vmovl_s16(vget_low_s16(b.v))},
      s32x4{vmovl_s16(vget_high_s16(b.v))},
  };
}

YNN_ALWAYS_INLINE s32x16 cast(s8x16 b, int32_t) {
  return cast(cast(b, int16_t{}), int32_t{});
}

YNN_ALWAYS_INLINE s32x16 cast(u8x16 b, int32_t) {
  return cast(cast(b, int16_t{}), int32_t{});
}

YNN_ALWAYS_INLINE f32x4 cast(s32x4 x, float) {
  return f32x4{vcvtq_f32_s32(x.v)};
}

#ifdef YNN_ARCH_ARM64
YNN_ALWAYS_INLINE f64x2 cast(f32x2 a, double) {
  return f64x2{vcvt_f64_f32(a.v)};
}
YNN_ALWAYS_INLINE f32x2 cast(f64x2 a, float) {
  return f32x2{vcvt_f32_f64(a.v)};
}
#endif  // YNN_ARCH_ARM64

YNN_ALWAYS_INLINE s16x8 cast(s32x8 a, int16_t) {
  return s16x8{vcombine_s16(vqmovn_s32(lo(a).v), vqmovn_s32(hi(a).v))};
}

YNN_ALWAYS_INLINE s8x16 cast(s16x16 a, int8_t) {
  return s8x16{vcombine_s8(vqmovn_s16(lo(a).v), vqmovn_s16(hi(a).v))};
}

YNN_ALWAYS_INLINE u8x16 cast(s16x16 a, uint8_t) {
  return u8x16{vcombine_u8(vqmovun_s16(lo(a).v), vqmovun_s16(hi(a).v))};
}

YNN_ALWAYS_INLINE s32x4 cast(f32x4 f, int32_t) {
#if defined(__ARM_ARCH) && __ARM_ARCH < 8
  return s32x4{vcvtq_s32_f32(round(f).v)};
#else
  return s32x4{vcvtnq_s32_f32(f.v)};
#endif
}

YNN_ALWAYS_INLINE s16x8 cast(f32x8 f, int16_t) {
#if defined(__ARM_ARCH) && __ARM_ARCH < 8
  s32x4 a1 = cast(round(lo(f)), int32_t{});
  s32x4 a2 = cast(round(hi(f)), int32_t{});
  return cast(s32x8{a1, a2}, int16_t{});
#else
  return s16x8{vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(lo(f).v)),
                            vqmovn_s32(vcvtnq_s32_f32(hi(f).v)))};
#endif
}

YNN_ALWAYS_INLINE s8x16 cast(f32x16 f, int8_t) {
  s16x16 f_s16 = {
      cast(lo(f), int16_t{}),
      cast(hi(f), int16_t{}),
  };
  return cast(f_s16, int8_t{});
}

YNN_ALWAYS_INLINE u8x16 cast(f32x16 f, uint8_t) {
  s16x16 f_s16 = {
      cast(lo(f), int16_t{}),
      cast(hi(f), int16_t{}),
  };
  return cast(f_s16, uint8_t{});
}

#ifdef YNN_ARCH_ARM_NEONFP16
using f32x8 = vec<float, 8>;

YNN_ALWAYS_INLINE f32x4 cast(f16x4 a, float) {
  return f32x4{vcvt_f32_f16(vreinterpret_f16_u16(a.v))};
}

YNN_ALWAYS_INLINE f32x8 cast(f16x8 a, float) {
  return {
      f32x4{vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(a.v)))},
      f32x4{vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(a.v)))},
  };
}

YNN_ALWAYS_INLINE f16x4 cast(f32x4 a, half) {
  return f16x4{vreinterpret_u16_f16(vcvt_f16_f32(a.v))};
}

YNN_ALWAYS_INLINE f16x8 cast(f32x8 a, half) {
  return f16x8{vreinterpretq_u16_f16(
      vcombine_f16(vcvt_f16_f32(lo(a).v), vcvt_f16_f32(hi(a).v)))};
}
#endif

}  // namespace simd

}  // namespace ynn

#include "ynnpack/base/simd/generic.inc"  // IWYU pragma: export

#endif  // XNNPACK_YNNPACK_BASE_SIMD_ARM_H_
