// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_WASM_SIMD128_H_
#define XNNPACK_YNNPACK_BASE_SIMD_WASM_SIMD128_H_

#include <wasm_simd128.h>

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

// See vec.h for architecture independent comments.

template <>
struct vec<int8_t, 16> {
  using value_type = int8_t;
  static constexpr std::integral_constant<size_t, 16> N = {};

  vec() = default;
  explicit vec(v128_t v) : v(v) {}
  vec(int8_t x) : v(wasm_i8x16_splat(x)) {}  // NOLINT

  v128_t v;
};

template <>
struct vec<uint8_t, 16> {
  using value_type = uint8_t;
  static constexpr std::integral_constant<size_t, 16> N = {};

  vec() = default;
  explicit vec(v128_t v) : v(v) {}
  vec(uint8_t x) : v(wasm_i8x16_splat(x)) {}  // NOLINT

  v128_t v;
};

template <>
struct vec<int16_t, 8> {
  using value_type = int16_t;
  static constexpr std::integral_constant<size_t, 8> N = {};

  vec() = default;
  explicit vec(v128_t v) : v(v) {}
  vec(int16_t x) : v(wasm_i16x8_splat(x)) {}  // NOLINT

  v128_t v;
};

template <>
struct vec<uint16_t, 8> {
  using value_type = uint16_t;
  static constexpr std::integral_constant<size_t, 8> N = {};

  vec() = default;
  explicit vec(v128_t v) : v(v) {}
  vec(uint16_t x) : v(wasm_i16x8_splat(x)) {}  // NOLINT

  v128_t v;
};

template <>
struct vec<int32_t, 4> {
  using value_type = int32_t;
  static constexpr std::integral_constant<size_t, 4> N = {};

  vec() = default;
  explicit vec(v128_t v) : v(v) {}
  vec(int32_t x) : v(wasm_i32x4_splat(x)) {}  // NOLINT

  v128_t v;
};

template <>
struct vec<uint32_t, 4> {
  using value_type = uint32_t;
  static constexpr std::integral_constant<size_t, 4> N = {};

  vec() = default;
  explicit vec(v128_t v) : v(v) {}
  vec(uint32_t x) : v(wasm_i32x4_splat(x)) {}  // NOLINT

  v128_t v;
};

template <>
struct vec<float, 4> {
  using value_type = float;
  static constexpr std::integral_constant<size_t, 4> N = {};

  vec() = default;
  explicit vec(v128_t v) : v(v) {}
  vec(float x) : v(wasm_f32x4_splat(x)) {}  // NOLINT

  v128_t v;
};

template <>
struct vec<double, 2> {
  using value_type = double;
  static constexpr std::integral_constant<size_t, 2> N = {};

  vec() = default;
  explicit vec(v128_t v) : v(v) {}
  vec(double x) : v(wasm_f64x2_splat(x)) {}  // NOLINT

  v128_t v;
};

using s8x16 = vec<int8_t, 16>;
using u8x16 = vec<uint8_t, 16>;
using s16x8 = vec<int16_t, 8>;
using u16x8 = vec<uint16_t, 8>;
using s32x4 = vec<int32_t, 4>;
using u32x4 = vec<uint32_t, 4>;
using f32x4 = vec<float, 4>;
using f64x2 = vec<double, 2>;

YNN_ALWAYS_INLINE u8x16 load(const uint8_t* ptr, decltype(u8x16::N),
                             u8x16 = {}) {
  return u8x16{wasm_v128_load(ptr)};
}
YNN_ALWAYS_INLINE s8x16 load(const int8_t* ptr, decltype(s8x16::N),
                             s8x16 = {}) {
  return s8x16{wasm_v128_load(ptr)};
}
YNN_ALWAYS_INLINE s16x8 load(const int16_t* ptr, decltype(s16x8::N),
                             s16x8 = {}) {
  return s16x8{wasm_v128_load(ptr)};
}
YNN_ALWAYS_INLINE u16x8 load(const uint16_t* ptr, decltype(u16x8::N),
                             u16x8 = {}) {
  return u16x8{wasm_v128_load(ptr)};
}
YNN_ALWAYS_INLINE s32x4 load(const int32_t* ptr, decltype(s32x4::N),
                             s32x4 = {}) {
  return s32x4{wasm_v128_load(ptr)};
}
YNN_ALWAYS_INLINE u32x4 load(const uint32_t* ptr, decltype(u32x4::N),
                             u32x4 = {}) {
  return u32x4{wasm_v128_load(ptr)};
}
YNN_ALWAYS_INLINE f32x4 load(const float* ptr, decltype(f32x4::N), f32x4 = {}) {
  return f32x4{wasm_v128_load(ptr)};
}
YNN_ALWAYS_INLINE f64x2 load(const double* ptr, decltype(f64x2::N),
                             f64x2 = {}) {
  return f64x2{wasm_v128_load(ptr)};
}

YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x16 b, decltype(u8x16::N) = {}) {
  wasm_v128_store(ptr, b.v);
}
YNN_ALWAYS_INLINE void store(int8_t* ptr, s8x16 b, decltype(s8x16::N) = {}) {
  wasm_v128_store(ptr, b.v);
}
YNN_ALWAYS_INLINE void store(int16_t* ptr, s16x8 b, decltype(s16x8::N) = {}) {
  wasm_v128_store(ptr, b.v);
}
YNN_ALWAYS_INLINE void store(uint16_t* ptr, u16x8 b, decltype(u16x8::N) = {}) {
  wasm_v128_store(ptr, b.v);
}
YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x4 b, decltype(s32x4::N) = {}) {
  wasm_v128_store(ptr, b.v);
}
YNN_ALWAYS_INLINE void store(uint32_t* ptr, u32x4 b, decltype(u32x4::N) = {}) {
  wasm_v128_store(ptr, b.v);
}
YNN_ALWAYS_INLINE void store(float* ptr, f32x4 b, decltype(f32x4::N) = {}) {
  wasm_v128_store(ptr, b.v);
}
YNN_ALWAYS_INLINE void store(double* ptr, f64x2 b, decltype(f64x2::N) = {}) {
  wasm_v128_store(ptr, b.v);
}

YNN_ALWAYS_INLINE f64x2 operator+(f64x2 a, f64x2 b) {
  return f64x2{wasm_f64x2_add(a.v, b.v)};
}
YNN_ALWAYS_INLINE f32x4 operator+(f32x4 a, f32x4 b) {
  return f32x4{wasm_f32x4_add(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x4 operator+(s32x4 a, s32x4 b) {
  return s32x4{wasm_i32x4_add(a.v, b.v)};
}
YNN_ALWAYS_INLINE u32x4 operator+(u32x4 a, u32x4 b) {
  return u32x4{wasm_i32x4_add(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 operator+(s16x8 a, s16x8 b) {
  return s16x8{wasm_i16x8_add(a.v, b.v)};
}
YNN_ALWAYS_INLINE u16x8 operator+(u16x8 a, u16x8 b) {
  return u16x8{wasm_i16x8_add(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 operator+(s8x16 a, s8x16 b) {
  return s8x16{wasm_i8x16_add(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 operator+(u8x16 a, u8x16 b) {
  return u8x16{wasm_i8x16_add(a.v, b.v)};
}

YNN_ALWAYS_INLINE f64x2 operator-(f64x2 a, f64x2 b) {
  return f64x2{wasm_f64x2_sub(a.v, b.v)};
}
YNN_ALWAYS_INLINE f32x4 operator-(f32x4 a, f32x4 b) {
  return f32x4{wasm_f32x4_sub(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x4 operator-(s32x4 a, s32x4 b) {
  return s32x4{wasm_i32x4_sub(a.v, b.v)};
}
YNN_ALWAYS_INLINE u32x4 operator-(u32x4 a, u32x4 b) {
  return u32x4{wasm_i32x4_sub(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 operator-(s16x8 a, s16x8 b) {
  return s16x8{wasm_i16x8_sub(a.v, b.v)};
}
YNN_ALWAYS_INLINE u16x8 operator-(u16x8 a, u16x8 b) {
  return u16x8{wasm_i16x8_sub(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 operator-(s8x16 a, s8x16 b) {
  return s8x16{wasm_i8x16_sub(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 operator-(u8x16 a, u8x16 b) {
  return u8x16{wasm_i8x16_sub(a.v, b.v)};
}

YNN_ALWAYS_INLINE s16x8 add_sat(s16x8 a, s16x8 b) {
  return s16x8{wasm_i16x8_add_sat(a.v, b.v)};
}
YNN_ALWAYS_INLINE u16x8 add_sat(u16x8 a, u16x8 b) {
  return u16x8{wasm_u16x8_add_sat(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 add_sat(s8x16 a, s8x16 b) {
  return s8x16{wasm_i8x16_add_sat(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 add_sat(u8x16 a, u8x16 b) {
  return u8x16{wasm_u8x16_add_sat(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 sub_sat(s16x8 a, s16x8 b) {
  return s16x8{wasm_i16x8_sub_sat(a.v, b.v)};
}
YNN_ALWAYS_INLINE u16x8 sub_sat(u16x8 a, u16x8 b) {
  return u16x8{wasm_u16x8_sub_sat(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 sub_sat(s8x16 a, s8x16 b) {
  return s8x16{wasm_i8x16_sub_sat(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 sub_sat(u8x16 a, u8x16 b) {
  return u8x16{wasm_u8x16_sub_sat(a.v, b.v)};
}

YNN_ALWAYS_INLINE f64x2 operator*(f64x2 a, f64x2 b) {
  return f64x2{wasm_f64x2_mul(a.v, b.v)};
}
YNN_ALWAYS_INLINE f32x4 operator*(f32x4 a, f32x4 b) {
  return f32x4{wasm_f32x4_mul(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x4 operator*(s32x4 a, s32x4 b) {
  return s32x4{wasm_i32x4_mul(a.v, b.v)};
}
YNN_ALWAYS_INLINE u32x4 operator*(u32x4 a, u32x4 b) {
  return u32x4{wasm_i32x4_mul(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 operator*(s16x8 a, s16x8 b) {
  return s16x8{wasm_i16x8_mul(a.v, b.v)};
}
YNN_ALWAYS_INLINE u16x8 operator*(u16x8 a, u16x8 b) {
  return u16x8{wasm_i16x8_mul(a.v, b.v)};
}

YNN_ALWAYS_INLINE f64x2 operator/(f64x2 a, f64x2 b) {
  return f64x2{wasm_f64x2_div(a.v, b.v)};
}
YNN_ALWAYS_INLINE f32x4 operator/(f32x4 a, f32x4 b) {
  return f32x4{wasm_f32x4_div(a.v, b.v)};
}

YNN_ALWAYS_INLINE s16x8 operator&(s16x8 a, s16x8 b) {
  return s16x8{wasm_v128_and(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 operator>>(s16x8 a, int b) {
  return s16x8{wasm_i16x8_shr(a.v, b)};
}
YNN_ALWAYS_INLINE s16x8 operator<<(s16x8 a, int b) {
  return s16x8{wasm_i16x8_shl(a.v, b)};
}
YNN_ALWAYS_INLINE s16x8 operator^(s16x8 a, s16x8 b) {
  return s16x8{wasm_v128_xor(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 operator|(s16x8 a, s16x8 b) {
  return s16x8{wasm_v128_or(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 operator~(s16x8 a) { return s16x8{wasm_v128_not(a.v)}; }

YNN_ALWAYS_INLINE s32x4 operator&(s32x4 a, s32x4 b) {
  return s32x4{wasm_v128_and(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x4 operator|(s32x4 a, s32x4 b) {
  return s32x4{wasm_v128_or(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x4 operator^(s32x4 a, s32x4 b) {
  return s32x4{wasm_v128_xor(a.v, b.v)};
}
YNN_ALWAYS_INLINE s32x4 operator~(s32x4 a) { return s32x4{wasm_v128_not(a.v)}; }
YNN_ALWAYS_INLINE s32x4 operator<<(s32x4 a, int b) {
  return s32x4{wasm_i32x4_shl(a.v, b)};
}
YNN_ALWAYS_INLINE u8x16 operator&(u8x16 a, u8x16 b) {
  return u8x16{wasm_v128_and(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 operator|(u8x16 a, u8x16 b) {
  return u8x16{wasm_v128_or(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 operator^(u8x16 a, u8x16 b) {
  return u8x16{wasm_v128_xor(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 operator~(u8x16 a) { return u8x16{wasm_v128_not(a.v)}; }

YNN_ALWAYS_INLINE s8x16 operator&(s8x16 a, s8x16 b) {
  return s8x16{wasm_v128_and(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 operator|(s8x16 a, s8x16 b) {
  return s8x16{wasm_v128_or(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 operator^(s8x16 a, s8x16 b) {
  return s8x16{wasm_v128_xor(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 operator~(s8x16 a) { return s8x16{wasm_v128_not(a.v)}; }

YNN_ALWAYS_INLINE f32x4 min(f32x4 a, f32x4 b) {
  return f32x4{wasm_f32x4_min(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 min(s16x8 a, s16x8 b) {
  return s16x8{wasm_i16x8_min(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 min(u8x16 a, u8x16 b) {
  return u8x16{wasm_u8x16_min(a.v, b.v)};
}

YNN_ALWAYS_INLINE f32x4 max(f32x4 a, f32x4 b) {
  return f32x4{wasm_f32x4_max(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 max(s16x8 a, s16x8 b) {
  return s16x8{wasm_i16x8_max(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 max(u8x16 a, u8x16 b) {
  return u8x16{wasm_u8x16_max(a.v, b.v)};
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_WASM_SIMD128_H_
