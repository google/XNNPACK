// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_ARM64_SVE_PARTIAL_LOAD_STORE_H_
#define XNNPACK_YNNPACK_BASE_SIMD_ARM64_SVE_PARTIAL_LOAD_STORE_H_

#include <arm_neon_sve_bridge.h>
#include <arm_sve.h>

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

namespace internal {

svbool_t mask_x32(uint32_t n) {
  return svwhilelt_b32(static_cast<uint32_t>(0), n);
}
svbool_t mask_x16(uint32_t n) {
  return svwhilelt_b16(static_cast<uint32_t>(0), n);
}
svbool_t mask_x8(uint32_t n) {
  return svwhilelt_b8(static_cast<uint32_t>(0), n);
}

svfloat32_t to_sve(float32x4_t x) { return svset_neonq(svundef_f32(), x); }
svint32_t to_sve(int32x4_t x) { return svset_neonq(svundef_s32(), x); }
svuint16_t to_sve(uint16x8_t x) { return svset_neonq(svundef_u16(), x); }
svint16_t to_sve(int16x8_t x) { return svset_neonq(svundef_s16(), x); }
svuint8_t to_sve(uint8x16_t x) { return svset_neonq(svundef_u8(), x); }
svint8_t to_sve(int8x16_t x) { return svset_neonq(svundef_s8(), x); }

svuint16_t to_sve(uint16x4_t x) {
  return svset_neonq(svundef_u16(), vcombine_u16(x, x));
}
svuint8_t to_sve(uint8x8_t x) {
  return svset_neonq(svundef_u8(), vcombine_u8(x, x));
}

}  // namespace internal

YNN_ALWAYS_INLINE f32x4 load(const float* ptr, size_t n, zeros<4> src) {
  return f32x4{svget_neonq(svld1(internal::mask_x32(n), ptr))};
}
YNN_ALWAYS_INLINE s32x4 load(const int32_t* ptr, size_t n, zeros<4> src) {
  return s32x4{svget_neonq(svld1(internal::mask_x32(n), ptr))};
}
YNN_ALWAYS_INLINE bf16x8 load(const bfloat16* ptr, size_t n, zeros<8> src) {
  return bf16x8{svget_neonq(
      svld1(internal::mask_x16(n), reinterpret_cast<const uint16_t*>(ptr)))};
}
YNN_ALWAYS_INLINE f16x8 load(const half* ptr, size_t n, zeros<8> src) {
  return f16x8{svget_neonq(
      svld1(internal::mask_x16(n), reinterpret_cast<const uint16_t*>(ptr)))};
}
YNN_ALWAYS_INLINE s16x8 load(const int16_t* ptr, size_t n, zeros<8> src) {
  return s16x8{svget_neonq(svld1(internal::mask_x16(n), ptr))};
}
YNN_ALWAYS_INLINE u8x16 load(const uint8_t* ptr, size_t n, zeros<16> src) {
  return u8x16{svget_neonq(svld1(internal::mask_x8(n), ptr))};
}
YNN_ALWAYS_INLINE s8x16 load(const int8_t* ptr, size_t n, zeros<16> src) {
  return s8x16{svget_neonq(svld1(internal::mask_x8(n), ptr))};
}

YNN_ALWAYS_INLINE f16x4 load(const half* ptr, size_t n, zeros<4> src) {
  return f16x4{vget_low_u16(svget_neonq(
      svld1(internal::mask_x16(n), reinterpret_cast<const uint16_t*>(ptr))))};
}
YNN_ALWAYS_INLINE u8x8 load(const uint8_t* ptr, size_t n, zeros<8> src) {
  return u8x8{vget_low_u8(svget_neonq(svld1(internal::mask_x8(n), ptr)))};
}

YNN_ALWAYS_INLINE f32x4 load(const float* ptr, size_t n, f32x4 src) {
  svbool_t m = internal::mask_x32(n);
  return f32x4{svget_neonq(svsel(m, svld1(m, ptr), internal::to_sve(src.v)))};
}
YNN_ALWAYS_INLINE s32x4 load(const int32_t* ptr, size_t n, s32x4 src) {
  svbool_t m = internal::mask_x32(n);
  return s32x4{svget_neonq(svsel(m, svld1(m, ptr), internal::to_sve(src.v)))};
}
YNN_ALWAYS_INLINE bf16x8 load(const bfloat16* ptr, size_t n, bf16x8 src) {
  svbool_t m = internal::mask_x16(n);
  return bf16x8{
      svget_neonq(svsel(m, svld1(m, reinterpret_cast<const uint16_t*>(ptr)),
                        internal::to_sve(src.v)))};
}
YNN_ALWAYS_INLINE f16x8 load(const half* ptr, size_t n, f16x8 src) {
  svbool_t m = internal::mask_x16(n);
  return f16x8{
      svget_neonq(svsel(m, svld1(m, reinterpret_cast<const uint16_t*>(ptr)),
                        internal::to_sve(src.v)))};
}
YNN_ALWAYS_INLINE s16x8 load(const int16_t* ptr, size_t n, s16x8 src) {
  svbool_t m = internal::mask_x16(n);
  return s16x8{svget_neonq(svsel(m, svld1(m, ptr), internal::to_sve(src.v)))};
}
YNN_ALWAYS_INLINE u8x16 load(const uint8_t* ptr, size_t n, u8x16 src) {
  svbool_t m = internal::mask_x8(n);
  return u8x16{svget_neonq(svsel(m, svld1(m, ptr), internal::to_sve(src.v)))};
}
YNN_ALWAYS_INLINE s8x16 load(const int8_t* ptr, size_t n, s8x16 src) {
  svbool_t m = internal::mask_x8(n);
  return s8x16{svget_neonq(svsel(m, svld1(m, ptr), internal::to_sve(src.v)))};
}

YNN_ALWAYS_INLINE f16x4 load(const half* ptr, size_t n, f16x4 src) {
  svbool_t m = internal::mask_x16(n);
  return f16x4{vget_low_u16(
      svget_neonq(svsel(m, svld1(m, reinterpret_cast<const uint16_t*>(ptr)),
                        internal::to_sve(src.v))))};
}
YNN_ALWAYS_INLINE u8x8 load(const uint8_t* ptr, size_t n, u8x8 src) {
  svbool_t m = internal::mask_x8(n);
  return u8x8{vget_low_u8(
      svget_neonq(svsel(m, svld1(m, ptr), internal::to_sve(src.v))))};
}

YNN_ALWAYS_INLINE f32x4 load(const float* ptr, size_t n, undef<4>) {
  return load(ptr, n, zeros<4>{});
}
YNN_ALWAYS_INLINE s32x4 load(const int32_t* ptr, size_t n, undef<4>) {
  return load(ptr, n, zeros<4>{});
}
YNN_ALWAYS_INLINE bf16x8 load(const bfloat16* ptr, size_t n, undef<8>) {
  return load(ptr, n, zeros<8>{});
}
YNN_ALWAYS_INLINE f16x8 load(const half* ptr, size_t n, undef<8>) {
  return load(ptr, n, zeros<8>{});
}
YNN_ALWAYS_INLINE s16x8 load(const int16_t* ptr, size_t n, undef<8>) {
  return load(ptr, n, zeros<8>{});
}
YNN_ALWAYS_INLINE u8x16 load(const uint8_t* ptr, size_t n, undef<16>) {
  return load(ptr, n, zeros<16>{});
}
YNN_ALWAYS_INLINE s8x16 load(const int8_t* ptr, size_t n, undef<16>) {
  return load(ptr, n, zeros<16>{});
}

YNN_ALWAYS_INLINE f16x4 load(const half* ptr, size_t n, undef<4>) {
  return load(ptr, n, zeros<4>{});
}
YNN_ALWAYS_INLINE u8x8 load(const uint8_t* ptr, size_t n, undef<8>) {
  return load(ptr, n, zeros<8>{});
}

YNN_ALWAYS_INLINE void store(float* ptr, f32x4 value, size_t n) {
  svst1(internal::mask_x32(n), ptr, internal::to_sve(value.v));
}
YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x4 value, size_t n) {
  svst1(internal::mask_x32(n), ptr, internal::to_sve(value.v));
}
YNN_ALWAYS_INLINE void store(bfloat16* ptr, bf16x8 value, size_t n) {
  svst1(internal::mask_x16(n), reinterpret_cast<uint16_t*>(ptr),
        internal::to_sve(value.v));
}
YNN_ALWAYS_INLINE void store(half* ptr, f16x8 value, size_t n) {
  svst1(internal::mask_x16(n), reinterpret_cast<uint16_t*>(ptr),
        internal::to_sve(value.v));
}
YNN_ALWAYS_INLINE void store(int16_t* ptr, s16x8 value, size_t n) {
  svst1(internal::mask_x16(n), ptr, internal::to_sve(value.v));
}
YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x16 value, size_t n) {
  svst1(internal::mask_x8(n), ptr, internal::to_sve(value.v));
}
YNN_ALWAYS_INLINE void store(int8_t* ptr, s8x16 value, size_t n) {
  svst1(internal::mask_x8(n), ptr, internal::to_sve(value.v));
}

YNN_ALWAYS_INLINE void store(half* ptr, f16x4 value, size_t n) {
  svst1(internal::mask_x16(n), reinterpret_cast<uint16_t*>(ptr),
        internal::to_sve(value.v));
}
YNN_ALWAYS_INLINE void store(uint8_t* ptr, u8x8 value, size_t n) {
  svst1(internal::mask_x8(n), ptr, internal::to_sve(value.v));
}

}  // namespace simd

}  // namespace ynn

#include "ynnpack/base/simd/generic.inc"  // IWYU pragma: export

#endif  // XNNPACK_YNNPACK_BASE_SIMD_ARM64_SVE_PARTIAL_LOAD_STORE_H_
