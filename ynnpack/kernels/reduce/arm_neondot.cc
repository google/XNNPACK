// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/simd/arm.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace simd {

// TODO: try get rid of
struct s32x4x2 {
  s32x4 v[2];

  using value_type = int32_t;
  static constexpr std::integral_constant<size_t, 8> N = {};

  s32x4x2() = default;
  explicit s32x4x2(int32_t x) : v{x, x} {};

  YNN_ALWAYS_INLINE s32x4x2 operator+(s32x4x2 a) const {
    s32x4x2 res;
    res.v[0] = this->v[0] + a.v[0];
    res.v[1] = this->v[1] + a.v[1];
    return res;
  }
};

struct s32x4x4 {
  s32x4x2 v[2];

  using value_type = int32_t;
  static constexpr std::integral_constant<size_t, 16> N = {};

  s32x4x4() = default;
  explicit s32x4x4(int32_t x) : v{s32x4x2(x), s32x4x2(x)} {};

  YNN_ALWAYS_INLINE s32x4x4 operator+(s32x4x4 a) const {
    s32x4x4 res;
    res.v[0] = this->v[0] + a.v[0];
    res.v[1] = this->v[1] + a.v[1];
    return res;
  }
};

static s32x4x4& operator+=(s32x4x4& a, s8x16 b) {
  int16x8_t b_lo = vmovl_s8(vget_low_s8(b.v));
  int16x8_t b_hi = vmovl_s8(vget_high_s8(b.v));

  s32x4 b_0(vmovl_s16(vget_low_s16(b_lo)));
  s32x4 b_1(vmovl_s16(vget_high_s16(b_lo)));
  s32x4 b_2(vmovl_s16(vget_low_s16(b_hi)));
  s32x4 b_3(vmovl_s16(vget_high_s16(b_hi)));

  a.v[0].v[0] += b_0;
  a.v[0].v[1] += b_1;
  a.v[1].v[0] += b_2;
  a.v[1].v[1] += b_3;

  return a;
}

static s32x4x4& operator+=(s32x4x4& a, u8x16 b) {
  int16x8_t b_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b.v)));
  int16x8_t b_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b.v)));

  s32x4 b_0(vmovl_s16(vget_low_s16(b_lo)));
  s32x4 b_1(vmovl_s16(vget_high_s16(b_lo)));
  s32x4 b_2(vmovl_s16(vget_low_s16(b_hi)));
  s32x4 b_3(vmovl_s16(vget_high_s16(b_hi)));

  a.v[0].v[0] += b_0;
  a.v[0].v[1] += b_1;
  a.v[1].v[0] += b_2;
  a.v[1].v[1] += b_3;

  return a;
}

static s32x4x2 load(const int32_t* ptr, s32x4x2, decltype(s32x4x2::N)) {
  s32x4x2 x;

  x.v[0] = load(ptr, s32x4{}, s32x4::N);
  x.v[1] = load(ptr + s32x4::N, s32x4{}, s32x4::N);

  return x;
}

static s32x4x2 load(const int32_t* ptr, s32x4x2, size_t n) {
  s32x4x2 x;

  if (n < s32x4::N) {
    x.v[0] = load(ptr, s32x4{}, n);
    x.v[1] = 0;
  } else {
    x.v[0] = load(ptr, s32x4{}, s32x4::N);
    x.v[1] = load(ptr + s32x4::N, s32x4{}, n - s32x4::N);
  }

  return x;
}

static s32x4x4 load(const int32_t* ptr, s32x4x4, decltype(s32x4x4::N)) {
  s32x4x4 x;

  x.v[0] = load(ptr, s32x4x2{}, s32x4x2::N);
  x.v[1] = load(ptr + s32x4x2::N, s32x4x2{}, s32x4x2::N);

  return x;
}

static s32x4x4 load(const int32_t* ptr, s32x4x4, size_t n) {
  s32x4x4 x;

  if (n < s32x4x2::N) {
    x.v[0] = load(ptr, s32x4x2{}, n);
    x.v[1] = s32x4x2(0);
  } else {
    x.v[0] = load(ptr, s32x4x2{}, s32x4x2::N);
    x.v[1] = load(ptr + s32x4x2::N, s32x4x2{}, n - s32x4x2::N);
  }

  return x;
}

static void store(int32_t* ptr, s32x4x2 b, decltype(s32x4x2::N)) {
  store(ptr, b.v[0]);
  store(ptr + s32x4::N, b.v[1], s32x4::N);
}

static void store(int32_t* ptr, s32x4x2 b, size_t n) {
  if (n < s32x4::N) {
    store(ptr, b.v[0], n);
  } else {
    store(ptr, b.v[0]);
    store(ptr + s32x4::N, b.v[1], n - s32x4::N);
  }
}

static void store(int32_t* ptr, s32x4x4 b, decltype(s32x4x4::N)) {
  store(ptr, b.v[0], s32x4x2::N);
  store(ptr + s32x4x2::N, b.v[1], s32x4x2::N);
}

static void store(int32_t* ptr, s32x4x4 b, size_t n) {
  if (n < s32x4x2::N) {
    store(ptr, b.v[0], n);
  } else {
    store(ptr, b.v[0], s32x4x2::N);
    store(ptr + s32x4x2::N, b.v[1], n - s32x4x2::N);
  }
}

static s32x4& reduce_add(
    s32x4& a, s8x16 b,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  a.v = vdotq_s32(a.v, b.v, vdupq_n_s8(1));
  return a;
}

// We want to accumulate uint8 dot products in int32 accumulators.
static s32x4& reduce_add(
    s32x4& a, u8x16 b,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  a.v = vreinterpretq_s32_u32(vdotq_u32(vreinterpretq_u32_s32(a.v), b.v,
                              vdupq_n_u8(1)));
  return a;
}

}  // namespace simd

using simd::s32x4;
using simd::s32x4x4;
using simd::s8x16;
using simd::u8x16;

void sum_int8_int32_4x16_neondot(size_t n, size_t k3, size_t k2, size_t k1,
                                 size_t a_stride_n, size_t a_stride_k3,
                                 size_t a_stride_k2, const void* a, size_t,
                                 void* c) {
  if (k1 == 1 && a_stride_n == sizeof(int8_t)) {
    tiled_reduce<sum_accumulator_k1_1<s8x16, s32x4x4>, int8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const int8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x4, 16>, int8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const int8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

void sum_uint8_int32_4x16_neondot(size_t n, size_t k3, size_t k2, size_t k1,
                                  size_t a_stride_n, size_t a_stride_k3,
                                  size_t a_stride_k2, const void* a, size_t,
                                  void* c) {
  if (k1 == 1 && a_stride_n == sizeof(uint8_t)) {
    tiled_reduce<sum_accumulator_k1_1<u8x16, s32x4x4>, uint8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x4, 16>, uint8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

}  // namespace ynn
