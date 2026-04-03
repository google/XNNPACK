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

using s32x16 = vec<int32_t, 16>;
using u32x16 = vec<uint32_t, 16>;
using f32x16 = vec<float, 16>;
using s16x16 = vec<int16_t, 16>;
using u16x16 = vec<uint16_t, 16>;
using s32x8 = vec<int32_t, 8>;
using u32x8 = vec<uint32_t, 8>;
using f32x8 = vec<float, 8>;

YNN_ALWAYS_INLINE f64x2 load_aligned(const double* ptr, decltype(f64x2::N),
                                     f64x2 = {}) {
  return f64x2{wasm_v128_load(ptr)};
}
YNN_ALWAYS_INLINE f32x4 load_aligned(const float* ptr, decltype(f32x4::N),
                                     f32x4 = {}) {
  return f32x4{wasm_v128_load(ptr)};
}
YNN_ALWAYS_INLINE u32x4 load_aligned(const uint32_t* ptr, decltype(u32x4::N),
                                     u32x4 = {}) {
  return u32x4{wasm_v128_load(ptr)};
}
YNN_ALWAYS_INLINE s32x4 load_aligned(const int32_t* ptr, decltype(s32x4::N),
                                     s32x4 = {}) {
  return s32x4{wasm_v128_load(ptr)};
}
YNN_ALWAYS_INLINE u16x8 load_aligned(const uint16_t* ptr, decltype(u16x8::N),
                                     u16x8 = {}) {
  return u16x8{wasm_v128_load(ptr)};
}
YNN_ALWAYS_INLINE s16x8 load_aligned(const int16_t* ptr, decltype(s16x8::N),
                                     s16x8 = {}) {
  return s16x8{wasm_v128_load(ptr)};
}
YNN_ALWAYS_INLINE u8x16 load_aligned(const uint8_t* ptr, decltype(u8x16::N),
                                     u8x16 = {}) {
  return u8x16{wasm_v128_load(ptr)};
}
YNN_ALWAYS_INLINE s8x16 load_aligned(const int8_t* ptr, decltype(s8x16::N),
                                     s8x16 = {}) {
  return s8x16{wasm_v128_load(ptr)};
}

YNN_ALWAYS_INLINE void store_aligned(double* ptr, f64x2 b,
                                     decltype(f64x2::N) = {}) {
  wasm_v128_store(ptr, b.v);
}
YNN_ALWAYS_INLINE void store_aligned(float* ptr, f32x4 b,
                                     decltype(f32x4::N) = {}) {
  wasm_v128_store(ptr, b.v);
}
YNN_ALWAYS_INLINE void store_aligned(uint32_t* ptr, u32x4 b,
                                     decltype(u32x4::N) = {}) {
  wasm_v128_store(ptr, b.v);
}
YNN_ALWAYS_INLINE void store_aligned(int32_t* ptr, s32x4 b,
                                     decltype(s32x4::N) = {}) {
  wasm_v128_store(ptr, b.v);
}
YNN_ALWAYS_INLINE void store_aligned(uint16_t* ptr, u16x8 b,
                                     decltype(u16x8::N) = {}) {
  wasm_v128_store(ptr, b.v);
}
YNN_ALWAYS_INLINE void store_aligned(int16_t* ptr, s16x8 b,
                                     decltype(s16x8::N) = {}) {
  wasm_v128_store(ptr, b.v);
}
YNN_ALWAYS_INLINE void store_aligned(uint8_t* ptr, u8x16 b,
                                     decltype(u8x16::N) = {}) {
  wasm_v128_store(ptr, b.v);
}
YNN_ALWAYS_INLINE void store_aligned(int8_t* ptr, s8x16 b,
                                     decltype(s8x16::N) = {}) {
  wasm_v128_store(ptr, b.v);
}

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

YNN_ALWAYS_INLINE f32x4 load(const float* ptr, size_t n, f32x4 src) {
  return internal::partial_load_memcpy(ptr, n, src);
}
YNN_ALWAYS_INLINE s32x4 load(const int32_t* ptr, size_t n, s32x4 src) {
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

YNN_ALWAYS_INLINE f32x4 load(const float* ptr, size_t n, zeros<4> src) {
  return internal::partial_load_memcpy(ptr, n, f32x4{0});
}
YNN_ALWAYS_INLINE s32x4 load(const int32_t* ptr, size_t n, zeros<4> src) {
  return internal::partial_load_memcpy(ptr, n, s32x4{0});
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

YNN_ALWAYS_INLINE f32x4 load(const float* ptr, size_t n, undef<4> src) {
  return internal::partial_load_memcpy(ptr, n, f32x4{});
}
YNN_ALWAYS_INLINE s32x4 load(const int32_t* ptr, size_t n, undef<4> src) {
  return internal::partial_load_memcpy(ptr, n, s32x4{});
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

YNN_ALWAYS_INLINE void store(float* ptr, f32x4 b, size_t n) {
  internal::partial_store_memcpy(ptr, b, n);
}
YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x4 b, size_t n) {
  internal::partial_store_memcpy(ptr, b, n);
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
YNN_ALWAYS_INLINE u16x8 min(u16x8 a, u16x8 b) {
  return u16x8{wasm_u16x8_min(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 min(s16x8 a, s16x8 b) {
  return s16x8{wasm_i16x8_min(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 min(u8x16 a, u8x16 b) {
  return u8x16{wasm_u8x16_min(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 min(s8x16 a, s8x16 b) {
  return s8x16{wasm_i8x16_min(a.v, b.v)};
}

YNN_ALWAYS_INLINE f32x4 max(f32x4 a, f32x4 b) {
  return f32x4{wasm_f32x4_max(a.v, b.v)};
}
YNN_ALWAYS_INLINE u16x8 max(u16x8 a, u16x8 b) {
  return u16x8{wasm_u16x8_max(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x8 max(s16x8 a, s16x8 b) {
  return s16x8{wasm_i16x8_max(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x16 max(u8x16 a, u8x16 b) {
  return u8x16{wasm_u8x16_max(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x16 max(s8x16 a, s8x16 b) {
  return s8x16{wasm_i8x16_max(a.v, b.v)};
}

YNN_ALWAYS_INLINE f32x4 abs(f32x4 a) { return f32x4{wasm_f32x4_abs(a.v)}; }
YNN_ALWAYS_INLINE u32x4 abs(s32x4 a) { return u32x4{wasm_i32x4_abs(a.v)}; }
YNN_ALWAYS_INLINE u16x8 abs(s16x8 a) { return u16x8{wasm_i16x8_abs(a.v)}; }
YNN_ALWAYS_INLINE u8x16 abs(s8x16 a) { return u8x16{wasm_i8x16_abs(a.v)}; }

YNN_ALWAYS_INLINE f32x4 floor(f32x4 a) { return f32x4{wasm_f32x4_floor(a.v)}; }
YNN_ALWAYS_INLINE f32x4 ceil(f32x4 a) { return f32x4{wasm_f32x4_ceil(a.v)}; }
YNN_ALWAYS_INLINE f32x4 round(f32x4 a) {
  return f32x4{wasm_f32x4_nearest(a.v)};
}
YNN_ALWAYS_INLINE f32x4 sqrt(f32x4 a) { return f32x4{wasm_f32x4_sqrt(a.v)}; }

YNN_ALWAYS_INLINE s16x16 cast(s8x16 a, int16_t) {
  return {s16x8{wasm_i16x8_extend_low_i8x16(a.v)},
          s16x8{wasm_i16x8_extend_high_i8x16(a.v)}};
}

YNN_ALWAYS_INLINE u16x16 cast(u8x16 a, uint16_t) {
  return {u16x8{wasm_u16x8_extend_low_u8x16(a.v)},
          u16x8{wasm_u16x8_extend_high_u8x16(a.v)}};
}

YNN_ALWAYS_INLINE s32x8 cast(s16x8 a, int32_t) {
  return {s32x4{wasm_i32x4_extend_low_i16x8(a.v)},
          s32x4{wasm_i32x4_extend_high_i16x8(a.v)}};
}

YNN_ALWAYS_INLINE s32x8 cast(u16x8 a, int32_t) {
  return {s32x4{wasm_u32x4_extend_low_u16x8(a.v)},
          s32x4{wasm_u32x4_extend_high_u16x8(a.v)}};
}

YNN_ALWAYS_INLINE s32x16 cast(s8x16 a, int32_t) {
  return cast(cast(a, int16_t()), int32_t());
}

YNN_ALWAYS_INLINE s32x16 cast(u8x16 a, int32_t) {
  return cast(cast(a, uint16_t()), int32_t());
}

YNN_ALWAYS_INLINE f32x4 cast(s32x4 x, float) {
  return f32x4{wasm_f32x4_convert_i32x4(x.v)};
}

YNN_ALWAYS_INLINE s32x4 cast(f32x4 x, int32_t) {
  return s32x4{wasm_i32x4_trunc_sat_f32x4(x.v)};
}

YNN_ALWAYS_INLINE s16x8 saturate_cast(s32x8 a, int16_t) {
  return s16x8{wasm_i16x8_narrow_i32x4(a.lo().v, a.hi().v)};
}

YNN_ALWAYS_INLINE s8x16 saturate_cast(s16x16 a, int8_t) {
  return s8x16{wasm_i8x16_narrow_i16x8(a.lo().v, a.hi().v)};
}

YNN_ALWAYS_INLINE u8x16 saturate_cast(s16x16 a, uint8_t) {
  return u8x16{wasm_u8x16_narrow_i16x8(a.lo().v, a.hi().v)};
}

YNN_ALWAYS_INLINE s16x8 round_float_to_int(f32x8 f, int16_t) {
  const v128_t i0 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(f.lo().v));
  const v128_t i1 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(f.hi().v));
  return saturate_cast(s32x8(s32x4(i0), s32x4(i1)), int16_t());
}

YNN_ALWAYS_INLINE s8x16 round_float_to_int(f32x16 f, int8_t) {
  const s16x8 i01 =
      round_float_to_int(f32x8(f.lo().lo(), f.lo().hi()), int16_t());
  const s16x8 i23 =
      round_float_to_int(f32x8(f.hi().lo(), f.hi().hi()), int16_t());
  return saturate_cast(s16x16(i01, i23), int8_t());
}

YNN_ALWAYS_INLINE u8x16 round_float_to_int(f32x16 f, uint8_t) {
  const v128_t i0 =
      wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(f.lo().lo().v));
  const v128_t i1 =
      wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(f.lo().hi().v));
  const v128_t i2 =
      wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(f.hi().lo().v));
  const v128_t i3 =
      wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(f.hi().hi().v));
  const v128_t i01_16 = wasm_i16x8_narrow_i32x4(i0, i1);
  const v128_t i23_16 = wasm_i16x8_narrow_i32x4(i2, i3);
  return u8x16{wasm_u8x16_narrow_i16x8(i01_16, i23_16)};
}

YNN_ALWAYS_INLINE int8_t horizontal_max(s8x16 a) {
  v128_t max =
      wasm_i8x16_max(a.v, wasm_v8x16_shuffle(a.v, a.v, 8, 9, 10, 11, 12, 13, 14,
                                             15, 0, 1, 2, 3, 4, 5, 6, 7));
  max = wasm_i8x16_max(max, wasm_v8x16_shuffle(max, max, 4, 5, 6, 7, 0, 1, 2, 3,
                                               12, 13, 14, 15, 8, 9, 10, 11));
  max = wasm_i8x16_max(max, wasm_v8x16_shuffle(max, max, 2, 3, 0, 1, 6, 7, 4, 5,
                                               10, 11, 8, 9, 14, 15, 12, 13));
  max = wasm_i8x16_max(max, wasm_v8x16_shuffle(max, max, 1, 0, 3, 2, 5, 4, 7, 6,
                                               9, 8, 11, 10, 13, 12, 15, 14));
  return wasm_i8x16_extract_lane(max, 0);
}
YNN_ALWAYS_INLINE uint8_t horizontal_max(u8x16 a) {
  v128_t max =
      wasm_u8x16_max(a.v, wasm_v8x16_shuffle(a.v, a.v, 8, 9, 10, 11, 12, 13, 14,
                                             15, 0, 1, 2, 3, 4, 5, 6, 7));
  max = wasm_u8x16_max(max, wasm_v8x16_shuffle(max, max, 4, 5, 6, 7, 0, 1, 2, 3,
                                               12, 13, 14, 15, 8, 9, 10, 11));
  max = wasm_u8x16_max(max, wasm_v8x16_shuffle(max, max, 2, 3, 0, 1, 6, 7, 4, 5,
                                               10, 11, 8, 9, 14, 15, 12, 13));
  max = wasm_u8x16_max(max, wasm_v8x16_shuffle(max, max, 1, 0, 3, 2, 5, 4, 7, 6,
                                               9, 8, 11, 10, 13, 12, 15, 14));
  return wasm_u8x16_extract_lane(max, 0);
}
YNN_ALWAYS_INLINE int16_t horizontal_max(s16x8 a) {
  v128_t max =
      wasm_i16x8_max(a.v, wasm_v8x16_shuffle(a.v, a.v, 8, 9, 10, 11, 12, 13, 14,
                                             15, 0, 1, 2, 3, 4, 5, 6, 7));
  max = wasm_i16x8_max(max, wasm_v8x16_shuffle(max, max, 4, 5, 6, 7, 0, 1, 2, 3,
                                               12, 13, 14, 15, 8, 9, 10, 11));
  max = wasm_i16x8_max(max, wasm_v8x16_shuffle(max, max, 2, 3, 0, 1, 6, 7, 4, 5,
                                               10, 11, 8, 9, 14, 15, 12, 13));
  return wasm_i16x8_extract_lane(max, 0);
}
YNN_ALWAYS_INLINE int32_t horizontal_max(s32x4 a) {
  v128_t max =
      wasm_i32x4_max(a.v, wasm_v8x16_shuffle(a.v, a.v, 8, 9, 10, 11, 12, 13, 14,
                                             15, 0, 1, 2, 3, 4, 5, 6, 7));
  max = wasm_i32x4_max(max, wasm_v8x16_shuffle(max, max, 4, 5, 6, 7, 0, 1, 2, 3,
                                               12, 13, 14, 15, 8, 9, 10, 11));
  return wasm_i32x4_extract_lane(max, 0);
}
YNN_ALWAYS_INLINE float horizontal_max(f32x4 a) {
  v128_t max =
      wasm_f32x4_max(a.v, wasm_v8x16_shuffle(a.v, a.v, 8, 9, 10, 11, 12, 13, 14,
                                             15, 0, 1, 2, 3, 4, 5, 6, 7));
  max = wasm_f32x4_max(max, wasm_v8x16_shuffle(max, max, 4, 5, 6, 7, 0, 1, 2, 3,
                                               12, 13, 14, 15, 8, 9, 10, 11));
  return wasm_f32x4_extract_lane(max, 0);
}
YNN_ALWAYS_INLINE int8_t horizontal_min(s8x16 a) {
  v128_t min =
      wasm_i8x16_min(a.v, wasm_v8x16_shuffle(a.v, a.v, 8, 9, 10, 11, 12, 13, 14,
                                             15, 0, 1, 2, 3, 4, 5, 6, 7));
  min = wasm_i8x16_min(min, wasm_v8x16_shuffle(min, min, 4, 5, 6, 7, 0, 1, 2, 3,
                                               12, 13, 14, 15, 8, 9, 10, 11));
  min = wasm_i8x16_min(min, wasm_v8x16_shuffle(min, min, 2, 3, 0, 1, 6, 7, 4, 5,
                                               10, 11, 8, 9, 14, 15, 12, 13));
  min = wasm_i8x16_min(min, wasm_v8x16_shuffle(min, min, 1, 0, 3, 2, 5, 4, 7, 6,
                                               9, 8, 11, 10, 13, 12, 15, 14));
  return wasm_i8x16_extract_lane(min, 0);
}
YNN_ALWAYS_INLINE uint8_t horizontal_min(u8x16 a) {
  v128_t min =
      wasm_u8x16_min(a.v, wasm_v8x16_shuffle(a.v, a.v, 8, 9, 10, 11, 12, 13, 14,
                                             15, 0, 1, 2, 3, 4, 5, 6, 7));
  min = wasm_u8x16_min(min, wasm_v8x16_shuffle(min, min, 4, 5, 6, 7, 0, 1, 2, 3,
                                               12, 13, 14, 15, 8, 9, 10, 11));
  min = wasm_u8x16_min(min, wasm_v8x16_shuffle(min, min, 2, 3, 0, 1, 6, 7, 4, 5,
                                               10, 11, 8, 9, 14, 15, 12, 13));
  min = wasm_u8x16_min(min, wasm_v8x16_shuffle(min, min, 1, 0, 3, 2, 5, 4, 7, 6,
                                               9, 8, 11, 10, 13, 12, 15, 14));
  return wasm_u8x16_extract_lane(min, 0);
}
YNN_ALWAYS_INLINE int16_t horizontal_min(s16x8 a) {
  v128_t min =
      wasm_i16x8_min(a.v, wasm_v8x16_shuffle(a.v, a.v, 8, 9, 10, 11, 12, 13, 14,
                                             15, 0, 1, 2, 3, 4, 5, 6, 7));
  min = wasm_i16x8_min(min, wasm_v8x16_shuffle(min, min, 4, 5, 6, 7, 0, 1, 2, 3,
                                               12, 13, 14, 15, 8, 9, 10, 11));
  min = wasm_i16x8_min(min, wasm_v8x16_shuffle(min, min, 2, 3, 0, 1, 6, 7, 4, 5,
                                               10, 11, 8, 9, 14, 15, 12, 13));
  return wasm_i16x8_extract_lane(min, 0);
}
YNN_ALWAYS_INLINE int32_t horizontal_min(s32x4 a) {
  v128_t min =
      wasm_i32x4_min(a.v, wasm_v8x16_shuffle(a.v, a.v, 8, 9, 10, 11, 12, 13, 14,
                                             15, 0, 1, 2, 3, 4, 5, 6, 7));
  min = wasm_i32x4_min(min, wasm_v8x16_shuffle(min, min, 4, 5, 6, 7, 0, 1, 2, 3,
                                               12, 13, 14, 15, 8, 9, 10, 11));
  return wasm_i32x4_extract_lane(min, 0);
}
YNN_ALWAYS_INLINE float horizontal_min(f32x4 a) {
  v128_t min =
      wasm_f32x4_min(a.v, wasm_v8x16_shuffle(a.v, a.v, 8, 9, 10, 11, 12, 13, 14,
                                             15, 0, 1, 2, 3, 4, 5, 6, 7));
  min = wasm_f32x4_min(min, wasm_v8x16_shuffle(min, min, 4, 5, 6, 7, 0, 1, 2, 3,
                                               12, 13, 14, 15, 8, 9, 10, 11));
  return wasm_f32x4_extract_lane(min, 0);
}

namespace internal {

// These are helpers for implementing interleave/transpose.
YNN_ALWAYS_INLINE v128_t unpacklo_x32x4(v128_t a, v128_t b) {
  return wasm_v32x4_shuffle(a, b, 0, 4, 1, 5);
}
YNN_ALWAYS_INLINE v128_t unpackhi_x32x4(v128_t a, v128_t b) {
  return wasm_v32x4_shuffle(a, b, 2, 6, 3, 7);
}

YNN_ALWAYS_INLINE v128_t unpacklo_x8x16(v128_t a, v128_t b) {
  return wasm_v8x16_shuffle(a, b, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6,
                            22, 7, 23);
}
YNN_ALWAYS_INLINE v128_t unpackhi_x8x16(v128_t a, v128_t b) {
  return wasm_v8x16_shuffle(a, b, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29,
                            14, 30, 15, 31);
}

YNN_ALWAYS_INLINE v128_t movehl(v128_t a, v128_t b) {
  return wasm_v32x4_shuffle(a, b, 6, 7, 2, 3);
}
YNN_ALWAYS_INLINE v128_t movelh(v128_t a, v128_t b) {
  return wasm_v32x4_shuffle(a, b, 0, 1, 4, 5);
}

}  // namespace internal

template <typename T>
YNN_ALWAYS_INLINE std::array<vec<T, 4>, 4> transpose(
    std::array<vec<T, 4>, 4> x) {
  vec<T, 4> t0{internal::unpacklo_x32x4(x[0].v, x[1].v)};
  vec<T, 4> t1{internal::unpacklo_x32x4(x[2].v, x[3].v)};
  vec<T, 4> t2{internal::unpackhi_x32x4(x[0].v, x[1].v)};
  vec<T, 4> t3{internal::unpackhi_x32x4(x[2].v, x[3].v)};
  return {{
      vec<T, 4>{internal::movelh(t0.v, t1.v)},
      vec<T, 4>{internal::movehl(t1.v, t0.v)},
      vec<T, 4>{internal::movelh(t2.v, t3.v)},
      vec<T, 4>{internal::movehl(t3.v, t2.v)},
  }};
}

}  // namespace simd

}  // namespace ynn

#include "ynnpack/base/simd/generic.inc"  // IWYU pragma: export

#endif  // XNNPACK_YNNPACK_BASE_SIMD_WASM_SIMD128_H_
