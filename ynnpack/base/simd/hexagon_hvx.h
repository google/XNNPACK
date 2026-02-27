// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_HEXAGON_HVX_H_
#define XNNPACK_YNNPACK_BASE_SIMD_HEXAGON_HVX_H_

#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/vec.h"

namespace ynn {

namespace simd {

namespace internal {

YNN_ALWAYS_INLINE uint32_t splat16x2(uint16_t x) { return x | (x << 16); }

YNN_ALWAYS_INLINE uint16_t splat8x2(uint8_t x) { return x | (x << 8); }

YNN_ALWAYS_INLINE uint32_t splat8x4(uint8_t x) {
  return splat16x2(splat8x2(x));
}

}  // namespace internal

// See vec.h for architecture independent comments.

template <>
struct vec<float, 32> {
  using value_type = float;
  static constexpr std::integral_constant<size_t, 32> N = {};

  vec() = default;
  explicit vec(HVX_Vector v) : v(v) {}
  vec(float x) : v(Q6_V_vsplat_R(bit_cast<uint32_t>(x))) {}  // NOLINT

  HVX_Vector v;
};

template <>
struct vec<bfloat16, 64> {
  using value_type = bfloat16;
  static constexpr std::integral_constant<size_t, 64> N = {};

  vec() = default;
  explicit vec(HVX_Vector v) : v(v) {}
  vec(bfloat16 x)
      : v(Q6_V_vsplat_R(internal::splat16x2(x.to_bits()))) {}  // NOLINT

  HVX_Vector v;
};

template <>
struct vec<half, 64> {
  using value_type = half;
  static constexpr std::integral_constant<size_t, 64> N = {};

  vec() = default;
  explicit vec(HVX_Vector v) : v(v) {}
  vec(half x) : v(Q6_V_vsplat_R(internal::splat16x2(x.to_bits()))) {}  // NOLINT

  HVX_Vector v;
};

template <>
struct vec<int16_t, 64> {
  using value_type = int16_t;
  static constexpr std::integral_constant<size_t, 64> N = {};

  vec() = default;
  explicit vec(HVX_Vector v) : v(v) {}
  vec(int16_t x) : v(Q6_V_vsplat_R(internal::splat16x2(x))) {}  // NOLINT

  HVX_Vector v;
};

template <>
struct vec<int32_t, 32> {
  using value_type = int32_t;
  static constexpr std::integral_constant<size_t, 32> N = {};

  vec() = default;
  explicit vec(HVX_Vector v) : v(v) {}
  vec(int16_t x) : v(Q6_V_vsplat_R(x)) {}  // NOLINT

  HVX_Vector v;
};

template <>
struct vec<uint8_t, 128> {
  using value_type = uint8_t;
  static constexpr std::integral_constant<size_t, 128> N = {};

  vec() = default;
  explicit vec(HVX_Vector v) : v(v) {}
  vec(int16_t x) : v(Q6_V_vsplat_R(internal::splat8x4(x))) {}  // NOLINT

  HVX_Vector v;
};

template <>
struct vec<int8_t, 128> {
  using value_type = int8_t;
  static constexpr std::integral_constant<size_t, 128> N = {};

  vec() = default;
  explicit vec(HVX_Vector v) : v(v) {}
  vec(int16_t x) : v(Q6_V_vsplat_R(internal::splat8x4(x))) {}  // NOLINT

  HVX_Vector v;
};

using f32x32 = vec<float, 32>;
using s32x32 = vec<int32_t, 32>;
using bf16x64 = vec<bfloat16, 64>;
using f16x64 = vec<half, 64>;
using s16x64 = vec<int16_t, 64>;
using u8x128 = vec<uint8_t, 128>;
using s8x128 = vec<int8_t, 128>;

namespace internal {

YNN_ALWAYS_INLINE HVX_Vector load_aligned(const void* ptr) {
  HVX_Vector result;
  memcpy(&result, reinterpret_cast<const HVX_Vector*>(ptr), sizeof(result));
  return result;
}

YNN_ALWAYS_INLINE void store_aligned(void* ptr, HVX_Vector v) {
  memcpy(reinterpret_cast<HVX_Vector*>(ptr), &v, sizeof(v));
}

YNN_ALWAYS_INLINE HVX_UVector load(const void* ptr) {
  HVX_UVector result;
  memcpy(&result, ptr, sizeof(result));
  return result;
}

YNN_ALWAYS_INLINE void store(void* ptr, HVX_UVector v) {
  memcpy(ptr, &v, sizeof(v));
}

}  // namespace internal

YNN_ALWAYS_INLINE f32x32 load_aligned(const float* ptr, decltype(f32x32::N),
                                      f32x32 = {}) {
  return f32x32{internal::load_aligned(ptr)};
}
YNN_ALWAYS_INLINE s32x32 load_aligned(const int32_t* ptr, decltype(s32x32::N),
                                      s32x32 = {}) {
  return s32x32{internal::load_aligned(ptr)};
}
YNN_ALWAYS_INLINE bf16x64 load_aligned(const bfloat16* ptr,
                                       decltype(bf16x64::N), bf16x64 = {}) {
  return bf16x64{internal::load_aligned(ptr)};
}
YNN_ALWAYS_INLINE f16x64 load_aligned(const half* ptr, decltype(f16x64::N),
                                      f16x64 = {}) {
  return f16x64{internal::load_aligned(ptr)};
}
YNN_ALWAYS_INLINE s16x64 load_aligned(const int16_t* ptr, decltype(s16x64::N),
                                      s16x64 = {}) {
  return s16x64{internal::load_aligned(ptr)};
}
YNN_ALWAYS_INLINE u8x128 load_aligned(const uint8_t* ptr, decltype(u8x128::N),
                                      u8x128 = {}) {
  return u8x128{internal::load_aligned(ptr)};
}
YNN_ALWAYS_INLINE s8x128 load_aligned(const int8_t* ptr, decltype(s8x128::N),
                                      s8x128 = {}) {
  return s8x128{internal::load_aligned(ptr)};
}

YNN_ALWAYS_INLINE void store_aligned(float* ptr, f32x32 b,
                                     decltype(f32x32::N) = {}) {
  internal::store_aligned(ptr, b.v);
}
YNN_ALWAYS_INLINE void store_aligned(bfloat16* ptr, bf16x64 b,
                                     decltype(bf16x64::N) = {}) {
  internal::store_aligned(ptr, b.v);
}
YNN_ALWAYS_INLINE void store_aligned(half* ptr, f16x64 b,
                                     decltype(f16x64::N) = {}) {
  internal::store_aligned(ptr, b.v);
}
YNN_ALWAYS_INLINE void store_aligned(int16_t* ptr, s16x64 b,
                                     decltype(s16x64::N) = {}) {
  internal::store_aligned(ptr, b.v);
}
YNN_ALWAYS_INLINE void store_aligned(int32_t* ptr, s32x32 b,
                                     decltype(s32x32::N) = {}) {
  internal::store_aligned(ptr, b.v);
}
YNN_ALWAYS_INLINE void store_aligned(uint8_t* ptr, u8x128 b,
                                     decltype(u8x128::N) = {}) {
  internal::store_aligned(ptr, b.v);
}
YNN_ALWAYS_INLINE void store_aligned(int8_t* ptr, s8x128 b,
                                     decltype(s8x128::N) = {}) {
  internal::store_aligned(ptr, b.v);
}

YNN_ALWAYS_INLINE f32x32 load(const float* ptr, decltype(f32x32::N),
                              f32x32 = {}) {
  return f32x32{internal::load(ptr)};
}
YNN_ALWAYS_INLINE s32x32 load(const int32_t* ptr, decltype(s32x32::N),
                              s32x32 = {}) {
  return s32x32{internal::load(ptr)};
}
YNN_ALWAYS_INLINE bf16x64 load(const bfloat16* ptr, decltype(bf16x64::N),
                               bf16x64 = {}) {
  return bf16x64{internal::load(ptr)};
}
YNN_ALWAYS_INLINE f16x64 load(const half* ptr, decltype(f16x64::N),
                              f16x64 = {}) {
  return f16x64{internal::load(ptr)};
}
YNN_ALWAYS_INLINE s16x64 load(const int16_t* ptr, decltype(s16x64::N),
                              s16x64 = {}) {
  return s16x64{internal::load(ptr)};
}
YNN_ALWAYS_INLINE u8x128 load(const uint8_t* ptr, decltype(u8x128::N),
                              u8x128 = {}) {
  return u8x128{internal::load(ptr)};
}
YNN_ALWAYS_INLINE s8x128 load(const int8_t* ptr, decltype(s8x128::N),
                              s8x128 = {}) {
  return s8x128{internal::load(ptr)};
}

YNN_ALWAYS_INLINE void store(float* ptr, f32x32 b, decltype(f32x32::N) = {}) {
  internal::store(ptr, b.v);
}
YNN_ALWAYS_INLINE void store(bfloat16* ptr, bf16x64 b,
                             decltype(bf16x64::N) = {}) {
  internal::store(ptr, b.v);
}
YNN_ALWAYS_INLINE void store(half* ptr, f16x64 b, decltype(f16x64::N) = {}) {
  internal::store(ptr, b.v);
}
YNN_ALWAYS_INLINE void store(int16_t* ptr, s16x64 b, decltype(s16x64::N) = {}) {
  internal::store(ptr, b.v);
}
YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x32 b, decltype(s32x32::N) = {}) {
  internal::store(ptr, b.v);
}
YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x128 b, decltype(u8x128::N) = {}) {
  internal::store(ptr, b.v);
}
YNN_ALWAYS_INLINE void store(int8_t* ptr, s8x128 b, decltype(s8x128::N) = {}) {
  internal::store(ptr, b.v);
}

// Partial load/store with a non-constant number of elements.

namespace internal {

// A partial load where the lanes past n are undefined values.
template <typename T>
YNN_ALWAYS_INLINE HVX_Vector partial_load(const T* ptr, size_t n) {
  assert(n > 0);
  // HVX aligned loads ignore the unaligned part of the address.
  HVX_Vector v0 = load_aligned(ptr);
  const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  const uintptr_t offset = addr & (sizeof(HVX_Vector) - 1);
  HVX_Vector v1 = offset + n * sizeof(T) > sizeof(HVX_Vector)
                      ? load_aligned(offset_bytes(ptr, sizeof(HVX_Vector)))
                      : v0;
  return Q6_V_valign_VVR(v1, v0, addr);
}

template <typename T>
YNN_ALWAYS_INLINE HVX_Vector partial_load(const T* ptr, size_t n,
                                          HVX_Vector src) {
  // Get the lanes we want to load.
  HVX_Vector result = partial_load(ptr, n);

  // Replace the out of bounds lanes with `src`.
  assert(n * sizeof(T) < 128);
  HVX_VectorPred mask = Q6_Q_vsetq_R(n * sizeof(T));
  return Q6_V_vmux_QVV(mask, result, src);
}

template <typename T>
YNN_ALWAYS_INLINE void partial_store(T* ptr, HVX_Vector v, size_t n) {
  const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  HVX_Vector value = Q6_V_vlalign_VVR(v, v, addr);
  uintptr_t offset = addr & (sizeof(HVX_Vector) - 1);

  HVX_VectorPred ql_not = Q6_Q_vsetq_R(addr);
  HVX_VectorPred qr = Q6_Q_vsetq2_R(offset + n * sizeof(T));

  if (offset + n * sizeof(T) > sizeof(HVX_Vector)) {
    Q6_vmem_QRIV(qr, offset_bytes(ptr, sizeof(HVX_Vector)), value);
    qr = Q6_Q_vcmp_eq_VbVb(value, value);
  }

  ql_not = Q6_Q_or_QQn(ql_not, qr);
  Q6_vmem_QnRIV(ql_not, ptr, value);
}

}  // namespace internal

YNN_ALWAYS_INLINE f32x32 load(const float* ptr, size_t n, f32x32 src) {
  return f32x32{internal::partial_load(ptr, n, src.v)};
}
YNN_ALWAYS_INLINE s32x32 load(const int32_t* ptr, size_t n, s32x32 src) {
  return s32x32{internal::partial_load(ptr, n, src.v)};
}
YNN_ALWAYS_INLINE bf16x64 load(const bfloat16* ptr, size_t n, bf16x64 src) {
  return bf16x64{internal::partial_load(ptr, n, src.v)};
}
YNN_ALWAYS_INLINE f16x64 load(const half* ptr, size_t n, f16x64 src) {
  return f16x64{internal::partial_load(ptr, n, src.v)};
}
YNN_ALWAYS_INLINE s16x64 load(const int16_t* ptr, size_t n, s16x64 src) {
  return s16x64{internal::partial_load(ptr, n, src.v)};
}
YNN_ALWAYS_INLINE u8x128 load(const uint8_t* ptr, size_t n, u8x128 src) {
  return u8x128{internal::partial_load(ptr, n, src.v)};
}
YNN_ALWAYS_INLINE s8x128 load(const int8_t* ptr, size_t n, s8x128 src) {
  return s8x128{internal::partial_load(ptr, n, src.v)};
}

YNN_ALWAYS_INLINE f32x32 load(const float* ptr, size_t n, zeros<32> src) {
  return f32x32{internal::partial_load(ptr, n, Q6_V_vzero())};
}
YNN_ALWAYS_INLINE s32x32 load(const int32_t* ptr, size_t n, zeros<32> src) {
  return s32x32{internal::partial_load(ptr, n, Q6_V_vzero())};
}
YNN_ALWAYS_INLINE bf16x64 load(const bfloat16* ptr, size_t n, zeros<64> src) {
  return bf16x64{internal::partial_load(ptr, n, Q6_V_vzero())};
}
YNN_ALWAYS_INLINE f16x64 load(const half* ptr, size_t n, zeros<64> src) {
  return f16x64{internal::partial_load(ptr, n, Q6_V_vzero())};
}
YNN_ALWAYS_INLINE s16x64 load(const int16_t* ptr, size_t n, zeros<64> src) {
  return s16x64{internal::partial_load(ptr, n, Q6_V_vzero())};
}
YNN_ALWAYS_INLINE u8x128 load(const uint8_t* ptr, size_t n, zeros<128> src) {
  return u8x128{internal::partial_load(ptr, n, Q6_V_vzero())};
}
YNN_ALWAYS_INLINE s8x128 load(const int8_t* ptr, size_t n, zeros<128> src) {
  return s8x128{internal::partial_load(ptr, n, Q6_V_vzero())};
}

YNN_ALWAYS_INLINE f32x32 load(const float* ptr, size_t n, undef<32> src) {
  return f32x32{internal::partial_load(ptr, n)};
}
YNN_ALWAYS_INLINE s32x32 load(const int32_t* ptr, size_t n, undef<32> src) {
  return s32x32{internal::partial_load(ptr, n)};
}
YNN_ALWAYS_INLINE bf16x64 load(const bfloat16* ptr, size_t n, undef<64> src) {
  return bf16x64{internal::partial_load(ptr, n)};
}
YNN_ALWAYS_INLINE f16x64 load(const half* ptr, size_t n, undef<64> src) {
  return f16x64{internal::partial_load(ptr, n)};
}
YNN_ALWAYS_INLINE s16x64 load(const int16_t* ptr, size_t n, undef<64> src) {
  return s16x64{internal::partial_load(ptr, n)};
}
YNN_ALWAYS_INLINE u8x128 load(const uint8_t* ptr, size_t n, undef<128> src) {
  return u8x128{internal::partial_load(ptr, n)};
}
YNN_ALWAYS_INLINE s8x128 load(const int8_t* ptr, size_t n, undef<128> src) {
  return s8x128{internal::partial_load(ptr, n)};
}

YNN_ALWAYS_INLINE void store(float* ptr, f32x32 b, size_t n) {
  internal::partial_store(ptr, b.v, n);
}
YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x32 b, size_t n) {
  internal::partial_store(ptr, b.v, n);
}
YNN_ALWAYS_INLINE void store(bfloat16* ptr, bf16x64 b, size_t n) {
  internal::partial_store(ptr, b.v, n);
}
YNN_ALWAYS_INLINE void store(half* ptr, f16x64 b, size_t n) {
  internal::partial_store(ptr, b.v, n);
}
YNN_ALWAYS_INLINE void store(int16_t* ptr, s16x64 b, size_t n) {
  internal::partial_store(ptr, b.v, n);
}
YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x128 b, size_t n) {
  internal::partial_store(ptr, b.v, n);
}
YNN_ALWAYS_INLINE void store(int8_t* ptr, s8x128 b, size_t n) {
  internal::partial_store(ptr, b.v, n);
}

YNN_ALWAYS_INLINE s32x32& operator+=(s32x32& a, s32x32 b) {
  a.v = Q6_Vw_vadd_VwVw(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s16x64& operator+=(s16x64& a, s16x64 b) {
  a.v = Q6_Vh_vadd_VhVh(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s8x128& operator+=(s8x128& a, s8x128 b) {
  a.v = Q6_Vb_vadd_VbVb(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE u8x128& operator+=(u8x128& a, u8x128 b) {
  a.v = Q6_Vb_vadd_VbVb(a.v, b.v);
  return a;
}

YNN_ALWAYS_INLINE s32x32& operator-=(s32x32& a, s32x32 b) {
  a.v = Q6_Vw_vsub_VwVw(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s16x64& operator-=(s16x64& a, s16x64 b) {
  a.v = Q6_Vh_vsub_VhVh(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE s8x128& operator-=(s8x128& a, s8x128 b) {
  a.v = Q6_Vb_vsub_VbVb(a.v, b.v);
  return a;
}
YNN_ALWAYS_INLINE u8x128& operator-=(u8x128& a, u8x128 b) {
  a.v = Q6_Vb_vsub_VbVb(a.v, b.v);
  return a;
}

YNN_ALWAYS_INLINE s32x32& operator*=(s32x32& a, s32x32 b) {
  // Hexagon doesn't have a 32-bit integer multiply, but it does have two
  // 32-bit x 16-bit multiply instructions that can be used to implement 32-bit
  // multiplication.
  HVX_Vector hi = Q6_Vw_vmpyieo_VhVh(a.v, b.v);
  a.v = Q6_Vw_vmpyieacc_VwVwVuh(hi, a.v, b.v);
  return a;
}

YNN_ALWAYS_INLINE s32x32 operator+(s32x32 a, s32x32 b) { return a += b; }
YNN_ALWAYS_INLINE s16x64 operator+(s16x64 a, s16x64 b) { return a += b; }
YNN_ALWAYS_INLINE s8x128 operator+(s8x128 a, s8x128 b) { return a += b; }
YNN_ALWAYS_INLINE u8x128 operator+(u8x128 a, u8x128 b) { return a += b; }

YNN_ALWAYS_INLINE s32x32 operator-(s32x32 a, s32x32 b) { return a -= b; }
YNN_ALWAYS_INLINE s16x64 operator-(s16x64 a, s16x64 b) { return a -= b; }
YNN_ALWAYS_INLINE s8x128 operator-(s8x128 a, s8x128 b) { return a -= b; }
YNN_ALWAYS_INLINE u8x128 operator-(u8x128 a, u8x128 b) { return a -= b; }

YNN_ALWAYS_INLINE s32x32 operator*(s32x32 a, s32x32 b) { return a *= b; }

YNN_ALWAYS_INLINE s16x64 operator&(s16x64 a, s16x64 b) {
  return s16x64{Q6_V_vand_VV(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x64 operator>>(s16x64 a, int b) {
  return s16x64{Q6_Vh_vasr_VhR(a.v, b)};
}
YNN_ALWAYS_INLINE s16x64 operator^(s16x64 a, s16x64 b) {
  return s16x64{Q6_V_vxor_VV(a.v, b.v)};
}

YNN_ALWAYS_INLINE f32x32 min(f32x32 a, f32x32 b) {
  return f32x32{Q6_Vsf_vmin_VsfVsf(a.v, b.v)};
}
YNN_ALWAYS_INLINE f16x64 min(f16x64 a, f16x64 b) {
  return f16x64{Q6_Vhf_vmin_VhfVhf(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x64 min(s16x64 a, s16x64 b) {
  return s16x64{Q6_Vh_vmin_VhVh(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x128 min(u8x128 a, u8x128 b) {
  return u8x128{Q6_Vub_vmin_VubVub(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x128 min(s8x128 a, s8x128 b) {
  return s8x128{Q6_Vb_vmin_VbVb(a.v, b.v)};
}

YNN_ALWAYS_INLINE f32x32 max(f32x32 a, f32x32 b) {
  return f32x32{Q6_Vsf_vmax_VsfVsf(a.v, b.v)};
}
YNN_ALWAYS_INLINE f16x64 max(f16x64 a, f16x64 b) {
  return f16x64{Q6_Vhf_vmax_VhfVhf(a.v, b.v)};
}
YNN_ALWAYS_INLINE s16x64 max(s16x64 a, s16x64 b) {
  return s16x64{Q6_Vh_vmax_VhVh(a.v, b.v)};
}
YNN_ALWAYS_INLINE u8x128 max(u8x128 a, u8x128 b) {
  return u8x128{Q6_Vub_vmax_VubVub(a.v, b.v)};
}
YNN_ALWAYS_INLINE s8x128 max(s8x128 a, s8x128 b) {
  return s8x128{Q6_Vb_vmax_VbVb(a.v, b.v)};
}

YNN_ALWAYS_INLINE int32_t horizontal_sum(s32x32 x) {
  x.v = Q6_Vw_vadd_VwVw(x.v, Q6_V_vror_VR(x.v, 64));
  x.v = Q6_Vw_vadd_VwVw(x.v, Q6_V_vror_VR(x.v, 32));
  x.v = Q6_Vw_vadd_VwVw(x.v, Q6_V_vror_VR(x.v, 16));
  x.v = Q6_Vw_vadd_VwVw(x.v, Q6_V_vror_VR(x.v, 8));
  x.v = Q6_Vw_vadd_VwVw(x.v, Q6_V_vror_VR(x.v, 4));
  return *(int32_t*)&x.v;
}

YNN_ALWAYS_INLINE float horizontal_min(f32x32 x) {
  x.v = Q6_Vsf_vmin_VsfVsf(x.v, Q6_V_vror_VR(x.v, 64));
  x.v = Q6_Vsf_vmin_VsfVsf(x.v, Q6_V_vror_VR(x.v, 32));
  x.v = Q6_Vsf_vmin_VsfVsf(x.v, Q6_V_vror_VR(x.v, 16));
  x.v = Q6_Vsf_vmin_VsfVsf(x.v, Q6_V_vror_VR(x.v, 8));
  x.v = Q6_Vsf_vmin_VsfVsf(x.v, Q6_V_vror_VR(x.v, 4));
  return *(float*)&x.v;
}
YNN_ALWAYS_INLINE half horizontal_min(f16x64 x) {
  x.v = Q6_Vhf_vmin_VhfVhf(x.v, Q6_V_vror_VR(x.v, 64));
  x.v = Q6_Vhf_vmin_VhfVhf(x.v, Q6_V_vror_VR(x.v, 32));
  x.v = Q6_Vhf_vmin_VhfVhf(x.v, Q6_V_vror_VR(x.v, 16));
  x.v = Q6_Vhf_vmin_VhfVhf(x.v, Q6_V_vror_VR(x.v, 8));
  x.v = Q6_Vhf_vmin_VhfVhf(x.v, Q6_V_vror_VR(x.v, 4));
  x.v = Q6_Vhf_vmin_VhfVhf(x.v, Q6_V_vror_VR(x.v, 2));
  return *(half*)&x.v;
}
YNN_ALWAYS_INLINE int32_t horizontal_min(s32x32 x) {
  x.v = Q6_Vw_vmin_VwVw(x.v, Q6_V_vror_VR(x.v, 64));
  x.v = Q6_Vw_vmin_VwVw(x.v, Q6_V_vror_VR(x.v, 32));
  x.v = Q6_Vw_vmin_VwVw(x.v, Q6_V_vror_VR(x.v, 16));
  x.v = Q6_Vw_vmin_VwVw(x.v, Q6_V_vror_VR(x.v, 8));
  x.v = Q6_Vw_vmin_VwVw(x.v, Q6_V_vror_VR(x.v, 4));
  return *(int32_t*)&x.v;
}
YNN_ALWAYS_INLINE int16_t horizontal_min(s16x64 x) {
  x.v = Q6_Vh_vmin_VhVh(x.v, Q6_V_vror_VR(x.v, 64));
  x.v = Q6_Vh_vmin_VhVh(x.v, Q6_V_vror_VR(x.v, 32));
  x.v = Q6_Vh_vmin_VhVh(x.v, Q6_V_vror_VR(x.v, 16));
  x.v = Q6_Vh_vmin_VhVh(x.v, Q6_V_vror_VR(x.v, 8));
  x.v = Q6_Vh_vmin_VhVh(x.v, Q6_V_vror_VR(x.v, 4));
  x.v = Q6_Vh_vmin_VhVh(x.v, Q6_V_vror_VR(x.v, 2));
  return *(int16_t*)&x.v;
}
YNN_ALWAYS_INLINE int8_t horizontal_min(s8x128 x) {
  x.v = Q6_Vb_vmin_VbVb(x.v, Q6_V_vror_VR(x.v, 64));
  x.v = Q6_Vb_vmin_VbVb(x.v, Q6_V_vror_VR(x.v, 32));
  x.v = Q6_Vb_vmin_VbVb(x.v, Q6_V_vror_VR(x.v, 16));
  x.v = Q6_Vb_vmin_VbVb(x.v, Q6_V_vror_VR(x.v, 8));
  x.v = Q6_Vb_vmin_VbVb(x.v, Q6_V_vror_VR(x.v, 4));
  x.v = Q6_Vb_vmin_VbVb(x.v, Q6_V_vror_VR(x.v, 2));
  x.v = Q6_Vb_vmin_VbVb(x.v, Q6_V_vror_VR(x.v, 1));
  return *(int8_t*)&x.v;
}
YNN_ALWAYS_INLINE uint8_t horizontal_min(u8x128 x) {
  x.v = Q6_Vub_vmin_VubVub(x.v, Q6_V_vror_VR(x.v, 64));
  x.v = Q6_Vub_vmin_VubVub(x.v, Q6_V_vror_VR(x.v, 32));
  x.v = Q6_Vub_vmin_VubVub(x.v, Q6_V_vror_VR(x.v, 16));
  x.v = Q6_Vub_vmin_VubVub(x.v, Q6_V_vror_VR(x.v, 8));
  x.v = Q6_Vub_vmin_VubVub(x.v, Q6_V_vror_VR(x.v, 4));
  x.v = Q6_Vub_vmin_VubVub(x.v, Q6_V_vror_VR(x.v, 2));
  x.v = Q6_Vub_vmin_VubVub(x.v, Q6_V_vror_VR(x.v, 1));
  return *(uint8_t*)&x.v;
}
YNN_ALWAYS_INLINE float horizontal_max(f32x32 x) {
  x.v = Q6_Vsf_vmax_VsfVsf(x.v, Q6_V_vror_VR(x.v, 64));
  x.v = Q6_Vsf_vmax_VsfVsf(x.v, Q6_V_vror_VR(x.v, 32));
  x.v = Q6_Vsf_vmax_VsfVsf(x.v, Q6_V_vror_VR(x.v, 16));
  x.v = Q6_Vsf_vmax_VsfVsf(x.v, Q6_V_vror_VR(x.v, 8));
  x.v = Q6_Vsf_vmax_VsfVsf(x.v, Q6_V_vror_VR(x.v, 4));
  return *(float*)&x.v;
}
YNN_ALWAYS_INLINE half horizontal_max(f16x64 x) {
  x.v = Q6_Vhf_vmax_VhfVhf(x.v, Q6_V_vror_VR(x.v, 64));
  x.v = Q6_Vhf_vmax_VhfVhf(x.v, Q6_V_vror_VR(x.v, 32));
  x.v = Q6_Vhf_vmax_VhfVhf(x.v, Q6_V_vror_VR(x.v, 16));
  x.v = Q6_Vhf_vmax_VhfVhf(x.v, Q6_V_vror_VR(x.v, 8));
  x.v = Q6_Vhf_vmax_VhfVhf(x.v, Q6_V_vror_VR(x.v, 4));
  x.v = Q6_Vhf_vmax_VhfVhf(x.v, Q6_V_vror_VR(x.v, 2));
  return *(half*)&x.v;
}
YNN_ALWAYS_INLINE int32_t horizontal_max(s32x32 x) {
  x.v = Q6_Vw_vmax_VwVw(x.v, Q6_V_vror_VR(x.v, 64));
  x.v = Q6_Vw_vmax_VwVw(x.v, Q6_V_vror_VR(x.v, 32));
  x.v = Q6_Vw_vmax_VwVw(x.v, Q6_V_vror_VR(x.v, 16));
  x.v = Q6_Vw_vmax_VwVw(x.v, Q6_V_vror_VR(x.v, 8));
  x.v = Q6_Vw_vmax_VwVw(x.v, Q6_V_vror_VR(x.v, 4));
  return *(int32_t*)&x.v;
}
YNN_ALWAYS_INLINE int16_t horizontal_max(s16x64 x) {
  x.v = Q6_Vh_vmax_VhVh(x.v, Q6_V_vror_VR(x.v, 64));
  x.v = Q6_Vh_vmax_VhVh(x.v, Q6_V_vror_VR(x.v, 32));
  x.v = Q6_Vh_vmax_VhVh(x.v, Q6_V_vror_VR(x.v, 16));
  x.v = Q6_Vh_vmax_VhVh(x.v, Q6_V_vror_VR(x.v, 8));
  x.v = Q6_Vh_vmax_VhVh(x.v, Q6_V_vror_VR(x.v, 4));
  x.v = Q6_Vh_vmax_VhVh(x.v, Q6_V_vror_VR(x.v, 2));
  return *(int16_t*)&x.v;
}
YNN_ALWAYS_INLINE int8_t horizontal_max(s8x128 x) {
  x.v = Q6_Vb_vmax_VbVb(x.v, Q6_V_vror_VR(x.v, 64));
  x.v = Q6_Vb_vmax_VbVb(x.v, Q6_V_vror_VR(x.v, 32));
  x.v = Q6_Vb_vmax_VbVb(x.v, Q6_V_vror_VR(x.v, 16));
  x.v = Q6_Vb_vmax_VbVb(x.v, Q6_V_vror_VR(x.v, 8));
  x.v = Q6_Vb_vmax_VbVb(x.v, Q6_V_vror_VR(x.v, 4));
  x.v = Q6_Vb_vmax_VbVb(x.v, Q6_V_vror_VR(x.v, 2));
  x.v = Q6_Vb_vmax_VbVb(x.v, Q6_V_vror_VR(x.v, 1));
  return *(int8_t*)&x.v;
}
YNN_ALWAYS_INLINE uint8_t horizontal_max(u8x128 x) {
  x.v = Q6_Vub_vmax_VubVub(x.v, Q6_V_vror_VR(x.v, 64));
  x.v = Q6_Vub_vmax_VubVub(x.v, Q6_V_vror_VR(x.v, 32));
  x.v = Q6_Vub_vmax_VubVub(x.v, Q6_V_vror_VR(x.v, 16));
  x.v = Q6_Vub_vmax_VubVub(x.v, Q6_V_vror_VR(x.v, 8));
  x.v = Q6_Vub_vmax_VubVub(x.v, Q6_V_vror_VR(x.v, 4));
  x.v = Q6_Vub_vmax_VubVub(x.v, Q6_V_vror_VR(x.v, 2));
  x.v = Q6_Vub_vmax_VubVub(x.v, Q6_V_vror_VR(x.v, 1));
  return *(uint8_t*)&x.v;
}

using s32x64 = simd::vec<int32_t, 64>;
using s32x128 = simd::vec<int32_t, 128>;
using s16x128 = simd::vec<int16_t, 128>;

YNN_ALWAYS_INLINE s16x128 convert(s8x128 x, int16_t) {
  HVX_VectorPair result = Q6_Wh_vunpack_Vb(x.v);
  return {s16x64{Q6_V_lo_W(result)}, s16x64{Q6_V_hi_W(result)}};
}
YNN_ALWAYS_INLINE s16x128 convert(u8x128 x, int16_t) {
  HVX_VectorPair result = Q6_Wuh_vunpack_Vub(x.v);
  return {s16x64{Q6_V_lo_W(result)}, s16x64{Q6_V_hi_W(result)}};
}

YNN_ALWAYS_INLINE s32x64 convert(s16x64 x, int32_t) {
  HVX_VectorPair result = Q6_Ww_vunpack_Vh(x.v);
  return {s32x32{Q6_V_lo_W(result)}, s32x32{Q6_V_hi_W(result)}};
}

YNN_ALWAYS_INLINE s32x128 convert(s8x128 x, int32_t) {
  HVX_VectorPair s16 = Q6_Wh_vunpack_Vb(x.v);
  return {convert(s16x64{Q6_V_lo_W(s16)}, int32_t{}),
          convert(s16x64{Q6_V_hi_W(s16)}, int32_t{})};
}
YNN_ALWAYS_INLINE s32x128 convert(u8x128 x, int32_t) {
  HVX_VectorPair s16 = Q6_Wuh_vunpack_Vub(x.v);
  return {convert(s16x64{Q6_V_lo_W(s16)}, int32_t{}),
          convert(s16x64{Q6_V_hi_W(s16)}, int32_t{})};
}

template <typename ElemSizeBits>
YNN_ALWAYS_INLINE std::tuple<u8x128, u8x128> interleave(
    ElemSizeBits elem_size_bits, u8x128 x0, u8x128 x1) {
  HVX_VectorPair x01 = Q6_W_vshuff_VVR(x1.v, x0.v, -(elem_size_bits / 8));
  return {u8x128{Q6_V_lo_W(x01)}, u8x128{Q6_V_hi_W(x01)}};
}

}  // namespace simd

}  // namespace ynn

#include "ynnpack/base/simd/generic.inc"  // IWYU pragma: export

#endif  // XNNPACK_YNNPACK_BASE_SIMD_HEXAGON_HVX_H_
