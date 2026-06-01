// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef XNNPACK_SRC_XNNPACK_SIMD_F16_WASMRELAXEDSIMD_H_
#define XNNPACK_SRC_XNNPACK_SIMD_F16_WASMRELAXEDSIMD_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <wasm_simd128.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"

// SIMD vector type for f16 using WASMRELAXEDSIMD.
typedef v128_t xnn_simd_f16_t;
#define xnn_simd_size_f16 8
#define xnn_simd_log2_size_f16 3
#define xnn_simd_bytes_f16 (xnn_simd_size_f16 * sizeof(uint16_t))

#define XNN_SIMD_HAS_NATIVE_FMA 1

#define XNN_SIMD_CONST_F16(var, val) \
  const xnn_simd_f16_t var = wasm_i16x8_splat(val);

#define XNN_SIMD_CONST_F16_FROM_INT16(var, val) \
  const xnn_simd_f16_t var = wasm_i16x8_splat(val);

#define XNN_SIMD_CONST_F16_FROM_FLOAT(var, val) \
  const xnn_simd_f16_t var = wasm_i16x8_splat(xnn_float16_to_bits(xnn_float16_from_float(val)));

// Arithmetic operations.
static XNN_INLINE xnn_simd_f16_t xnn_zero_f16() {
  return wasm_i16x8_const_splat(0);
}

static XNN_INLINE xnn_simd_f16_t xnn_add_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return wasm_f16x8_add(a, b);
}

static XNN_INLINE xnn_simd_f16_t xnn_mul_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return wasm_f16x8_mul(a, b);
}

static XNN_INLINE xnn_simd_f16_t xnn_fmadd_f16(xnn_simd_f16_t a,
                                               xnn_simd_f16_t b,
                                               xnn_simd_f16_t c) {
  return wasm_f16x8_relaxed_madd(a, b, c);
}

static XNN_INLINE xnn_simd_f16_t xnn_fnmadd_f16(xnn_simd_f16_t a,
                                                xnn_simd_f16_t b,
                                                xnn_simd_f16_t c) {
  return wasm_f16x8_relaxed_nmadd(a, b, c);
}

static XNN_INLINE xnn_simd_f16_t xnn_fmsub_f16(xnn_simd_f16_t a,
                                               xnn_simd_f16_t b,
                                               xnn_simd_f16_t c) {
  return wasm_f16x8_relaxed_madd(a, b, wasm_f16x8_neg(c));
}

static XNN_INLINE xnn_simd_f16_t xnn_sub_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return wasm_f16x8_sub(a, b);
}

static XNN_INLINE xnn_simd_f16_t xnn_div_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return wasm_f16x8_div(a, b);
}

static XNN_INLINE xnn_simd_f16_t xnn_rcp_f16(xnn_simd_f16_t a) {
  return wasm_f16x8_div(wasm_i16x8_const_splat(0x3C00), a);
}

static XNN_INLINE xnn_simd_f16_t xnn_max_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return wasm_f16x8_pmax(a, b);
}

static XNN_INLINE xnn_simd_f16_t xnn_min_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return wasm_f16x8_pmin(a, b);
}

static XNN_INLINE xnn_simd_f16_t xnn_abs_f16(xnn_simd_f16_t a) {
  return wasm_f16x8_abs(a);
}

static XNN_INLINE xnn_simd_f16_t xnn_sqrt_f16(xnn_simd_f16_t a) {
  return wasm_f16x8_sqrt(a);
}

static XNN_INLINE xnn_simd_f16_t xnn_neg_f16(xnn_simd_f16_t a) {
  return wasm_f16x8_neg(a);
}

static XNN_INLINE xnn_simd_f16_t xnn_round_f16(xnn_simd_f16_t a) {
  return wasm_f16x8_nearest(a);
}

// Logical operations.
static XNN_INLINE xnn_simd_f16_t xnn_and_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return wasm_v128_and(a, b);
}

static XNN_INLINE xnn_simd_f16_t xnn_or_f16(xnn_simd_f16_t a,
                                            xnn_simd_f16_t b) {
  return wasm_v128_or(a, b);
}

static XNN_INLINE xnn_simd_f16_t xnn_xor_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return wasm_v128_xor(a, b);
}

static XNN_INLINE xnn_simd_f16_t xnn_sll_f16(xnn_simd_f16_t a, uint8_t bits) {
  return wasm_i16x8_shl(a, bits);
}

static XNN_INLINE xnn_simd_f16_t xnn_srl_f16(xnn_simd_f16_t a, uint8_t bits) {
  return wasm_u16x8_shr(a, bits);
}

static XNN_INLINE xnn_simd_f16_t xnn_sra_f16(xnn_simd_f16_t a, uint8_t bits) {
  return wasm_i16x8_shr(a, bits);
}

// TODO: use wasm_f16x8_eq once bug is fixed.
static XNN_INLINE xnn_simd_f16_t xnn_cmpeq_f16(xnn_simd_f16_t a,
                                               xnn_simd_f16_t b) {
  const xnn_simd_f16_t abs_mask = wasm_i16x8_splat(0x7FFF);
  const xnn_simd_f16_t inf = wasm_i16x8_splat(0x7C00);
  const xnn_simd_f16_t zero = wasm_i16x8_splat(0);

  const xnn_simd_f16_t abs_a = wasm_v128_and(a, abs_mask);
  const xnn_simd_f16_t abs_b = wasm_v128_and(b, abs_mask);

  const xnn_simd_f16_t bitwise_eq = wasm_i16x8_eq(a, b);
  const xnn_simd_f16_t both_zero =
      wasm_v128_and(wasm_i16x8_eq(abs_a, zero), wasm_i16x8_eq(abs_b, zero));

  const xnn_simd_f16_t eq = wasm_v128_or(bitwise_eq, both_zero);
  const xnn_simd_f16_t a_is_not_nan = wasm_i16x8_le(abs_a, inf);
  const xnn_simd_f16_t b_is_not_nan = wasm_i16x8_le(abs_b, inf);

  return wasm_v128_and(eq, wasm_v128_and(a_is_not_nan, b_is_not_nan));
}

// Load/store operations.
static XNN_INLINE xnn_simd_f16_t xnn_loadu_f16(const xnn_float16* ptr) {
  return wasm_v128_load(ptr);
}

static XNN_INLINE xnn_simd_f16_t xnn_load_f16(const xnn_float16* ptr) {
  return wasm_v128_load(ptr);
}

static XNN_INLINE void xnn_storeu_f16(xnn_float16* ptr, xnn_simd_f16_t v) {
  wasm_v128_store(ptr, v);
}

static XNN_INLINE void xnn_store_f16(xnn_float16* ptr, xnn_simd_f16_t v) {
  wasm_v128_store(ptr, v);
}

static XNN_INLINE xnn_simd_f16_t xnn_set1_f16(xnn_float16 v) {
  return wasm_i16x8_splat(v.value);
}

// Tail load/store operations.
static XNN_INLINE xnn_simd_f16_t
xnn_load_tail_f16(const xnn_float16* input, size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_f16);

  XNN_ALIGN(16) xnn_float16 padded[8] = {0};
  xnn_float16* dst = padded;
  switch (num_elements) {
    case 7:
      *dst++ = *input++;
      XNN_FALLTHROUGH
    case 6:
      *dst++ = *input++;
      XNN_FALLTHROUGH
    case 5:
      *dst++ = *input++;
      XNN_FALLTHROUGH
    case 4:
      *dst++ = *input++;
      XNN_FALLTHROUGH
    case 3:
      *dst++ = *input++;
      XNN_FALLTHROUGH
    case 2:
      *dst++ = *input++;
      XNN_FALLTHROUGH
    default:
      *dst++ = *input++;
      break;
  }
  return wasm_v128_load(padded);
}

static XNN_INLINE void xnn_store_tail_f16(xnn_float16* output, xnn_simd_f16_t v,
                                          size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_f16);

  if (num_elements & 4) {
    wasm_v128_store64_lane(output, v, 0);
    v = wasm_v64x2_shuffle(v, v, 1, 1);
    output += 4;
  }
  if (num_elements & 2) {
    wasm_v128_store32_lane(output, v, 0);
    v = wasm_v32x4_shuffle(v, v, 1, 1, 1, 1);
    output += 2;
  }
  if (num_elements & 1) {
    wasm_v128_store16_lane(output, v, 0);
  }
}

// Conversion operations.
// TODO: Remove __builtin_convertvector once the wasm header is fixed.
typedef __fp16 xnn_f16x4 __attribute__((__vector_size__(8), __aligned__(8)));

static XNN_INLINE v128_t xnn_wasm_f32x4_promote_low_f16x8(v128_t __a) {
  return (v128_t) __builtin_convertvector(
      (xnn_f16x4){((__f16x8)__a)[0], ((__f16x8)__a)[1], ((__f16x8)__a)[2],
                  ((__f16x8)__a)[3]},
      __f32x4);
}

static XNN_INLINE v128_t xnn_wasm_f16x8_demote_f32x4_zero(v128_t __a) {
  return (v128_t) __builtin_convertvector(
      __builtin_shufflevector((__f32x4)__a, (__f32x4){0.0f, 0.0f, 0.0f, 0.0f},
                              0, 1, 2, 3, 4, 5, 6, 7),
      __f16x8);
}

static XNN_INLINE v128_t xnn_cvt_f32_f16(v128_t h) {
  return xnn_wasm_f32x4_promote_low_f16x8(h);
}

static XNN_INLINE v128_t xnn_cvt_f16_f32(v128_t f) {
  return xnn_wasm_f16x8_demote_f32x4_zero(f);
}

#endif  // XNNPACK_SRC_XNNPACK_SIMD_F16_WASMRELAXEDSIMD_H_
