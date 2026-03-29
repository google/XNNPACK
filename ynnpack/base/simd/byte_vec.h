// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_BYTE_VEC_H_
#define XNNPACK_YNNPACK_BASE_SIMD_BYTE_VEC_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <type_traits>

#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/vec.h"  // IWYU pragma: export

namespace ynn {

namespace simd {

// See vec.h for architecture independent comments.

// This implementation of `vec` uses integers to implement vectors of smaller
// integers. It does not support most arithmetic operations.

template <>
struct vec<uint8_t, 4> {
  using value_type = uint8_t;
  static constexpr std::integral_constant<size_t, 4> N = {};

  vec() = default;
  explicit vec(uint32_t v) : v(v) {}

  uint32_t v;
};

using u8x4 = vec<uint8_t, 4>;

template <>
struct vec<uint8_t, 8> {
  using value_type = uint8_t;
  static constexpr std::integral_constant<size_t, 8> N = {};

  vec() = default;
  explicit vec(uint64_t v) : v(v) {}
  vec(u8x4 x0, u8x4 x1) : v((static_cast<uint64_t>(x1.v) << 32) | x0.v) {}

  u8x4 lo() const { return u8x4{static_cast<uint32_t>(v)}; }
  u8x4 hi() const { return u8x4{static_cast<uint32_t>(v >> 32)}; }

  uint64_t v;
};

using u8x8 = vec<uint8_t, 8>;

YNN_ALWAYS_INLINE u8x4 load_aligned(const uint8_t* ptr, decltype(u8x4::N),
                                    u8x4 = {}) {
  return u8x4{*reinterpret_cast<const uint32_t*>(ptr)};
}
YNN_ALWAYS_INLINE u8x8 load_aligned(const uint8_t* ptr, decltype(u8x8::N),
                                    u8x8 = {}) {
  return u8x8{*reinterpret_cast<const uint64_t*>(ptr)};
}

YNN_ALWAYS_INLINE void store_aligned(uint8_t* ptr, u8x4 b,
                                     decltype(u8x4::N) = {}) {
  *reinterpret_cast<uint32_t*>(ptr) = b.v;
}
YNN_ALWAYS_INLINE void store_aligned(uint8_t* ptr, u8x8 b,
                                     decltype(u8x8::N) = {}) {
  *reinterpret_cast<uint64_t*>(ptr) = b.v;
}

YNN_ALWAYS_INLINE u8x4 load(const uint8_t* ptr, decltype(u8x4::N), u8x4 = {}) {
  uint32_t mem;
  memcpy(&mem, ptr, sizeof(mem));
  return u8x4{mem};
}
YNN_ALWAYS_INLINE u8x8 load(const uint8_t* ptr, decltype(u8x8::N), u8x8 = {}) {
  uint64_t mem;
  memcpy(&mem, ptr, sizeof(mem));
  return u8x8{mem};
}

YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x4 b, decltype(u8x4::N) = {}) {
  memcpy(ptr, &b, sizeof(b));
}
YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x8 b, decltype(u8x8::N) = {}) {
  memcpy(ptr, &b, sizeof(b));
}

YNN_ALWAYS_INLINE u8x4 load(const uint8_t* ptr, size_t n, u8x4 src) {
  switch (n) {
    case 4:
      memcpy(&src, ptr, 4);
      break;
    case 3:
      memcpy(&src, ptr, 3);
      break;
    case 2:
      memcpy(&src, ptr, 2);
      break;
    case 1:
      memcpy(&src, ptr, 1);
      break;
    default:
      break;
  }
  return src;
}
YNN_ALWAYS_INLINE u8x4 load(const uint8_t* ptr, size_t n, zeros<4> src) {
  return load(ptr, n, u8x4{0});
}
YNN_ALWAYS_INLINE u8x4 load(const uint8_t* ptr, size_t n, undef<4> src) {
  return load(ptr, n, u8x4{0});
}
YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x4 val, size_t n) {
  switch (n) {
    case 4:
      memcpy(ptr, &val, 4);
      break;
    case 3:
      memcpy(ptr, &val, 3);
      break;
    case 2:
      memcpy(ptr, &val, 2);
      break;
    case 1:
      memcpy(ptr, &val, 1);
      break;
    default:
      break;
  }
}

YNN_ALWAYS_INLINE std::tuple<u8x4, u8x4> interleave(
    std::integral_constant<size_t, 16>, u8x4 x0, u8x4 x1) {
  return {u8x4{(x0.v & 0xFFFF) | (x1.v << 16)},
          u8x4{(x0.v >> 16) | (x1.v & 0xFFFF0000)}};
}
YNN_ALWAYS_INLINE std::tuple<u8x4, u8x4> interleave(
    std::integral_constant<size_t, 8>, u8x4 x0, u8x4 x1) {
  constexpr uint32_t m0 = 0x00FF00FF;
  constexpr uint32_t m1 = 0xFF00FF00;
  u8x4 t0{(x0.v & m0) | ((x1.v & m0) << 8)};
  u8x4 t1{((x0.v & m1) >> 8) | (x1.v & m1)};
  return interleave(std::integral_constant<size_t, 16>{}, t0, t1);
}
YNN_ALWAYS_INLINE std::tuple<u8x4, u8x4> interleave(
    std::integral_constant<size_t, 4>, u8x4 x0, u8x4 x1) {
  constexpr uint32_t m0 = 0x0F0F0F0F;
  constexpr uint32_t m1 = 0xF0F0F0F0;
  u8x4 t0{(x0.v & m0) | ((x1.v & m0) << 4)};
  u8x4 t1{((x0.v & m1) >> 4) | (x1.v & m1)};
  return interleave(std::integral_constant<size_t, 8>{}, t0, t1);
}
YNN_ALWAYS_INLINE std::tuple<u8x4, u8x4> interleave(
    std::integral_constant<size_t, 2>, u8x4 x0, u8x4 x1) {
  constexpr uint32_t m0 = 0x33333333;
  constexpr uint32_t m1 = 0xCCCCCCCC;
  u8x4 t0{(x0.v & m0) | ((x1.v & m0) << 2)};
  u8x4 t1{((x0.v & m1) >> 2) | (x1.v & m1)};
  return interleave(std::integral_constant<size_t, 4>{}, t0, t1);
}

YNN_ALWAYS_INLINE std::tuple<u8x8, u8x8> interleave(
    std::integral_constant<size_t, 32>, u8x8 x0, u8x8 x1) {
  return {u8x8{(x0.v & 0xFFFFFFFF) | (x1.v << 32)},
          u8x8{(x0.v >> 32) | (x1.v & 0xFFFFFFFF00000000)}};
}
YNN_ALWAYS_INLINE std::tuple<u8x8, u8x8> interleave(
    std::integral_constant<size_t, 16>, u8x8 x0, u8x8 x1) {
  constexpr uint64_t m0 = 0x0000FFFF0000FFFF;
  constexpr uint64_t m1 = 0xFFFF0000FFFF0000;
  u8x8 t0{(x0.v & m0) | ((x1.v & m0) << 8)};
  u8x8 t1{((x0.v & m1) >> 8) | (x1.v & m1)};
  return interleave(std::integral_constant<size_t, 32>{}, t0, t1);
}
YNN_ALWAYS_INLINE std::tuple<u8x8, u8x8> interleave(
    std::integral_constant<size_t, 8>, u8x8 x0, u8x8 x1) {
  constexpr uint64_t m0 = 0x00FF00FF00FF00FF;
  constexpr uint64_t m1 = 0xFF00FF00FF00FF00;
  u8x8 t0{(x0.v & m0) | ((x1.v & m0) << 4)};
  u8x8 t1{((x0.v & m1) >> 4) | (x1.v & m1)};
  return interleave(std::integral_constant<size_t, 16>{}, t0, t1);
}
YNN_ALWAYS_INLINE std::tuple<u8x8, u8x8> interleave(
    std::integral_constant<size_t, 4>, u8x8 x0, u8x8 x1) {
  constexpr uint64_t m0 = 0x0F0F0F0F0F0F0F0F;
  constexpr uint64_t m1 = 0xF0F0F0F0F0F0F0F0;
  u8x8 t0{(x0.v & m0) | ((x1.v & m0) << 2)};
  u8x8 t1{((x0.v & m1) >> 2) | (x1.v & m1)};
  return interleave(std::integral_constant<size_t, 8>{}, t0, t1);
}

}  // namespace simd

}  // namespace ynn

#include "ynnpack/base/simd/generic.inc"  // IWYU pragma: export

#endif  // XNNPACK_YNNPACK_BASE_SIMD_BYTE_VEC_H_
