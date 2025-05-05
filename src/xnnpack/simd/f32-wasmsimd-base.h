// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_F32_WASMSIMD_BASE_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_F32_WASMSIMD_BASE_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <wasm_simd128.h>

#include "src/xnnpack/common.h"

// SIMD vector type for f32 using WASMSIMD.
typedef v128_t xnn_simd_f32_t;
#define xnn_simd_size_f32 4
#define xnn_simd_log2_size_f32 2
#define xnn_simd_bytes_f32 (xnn_simd_size_f32 * sizeof(float))

#define XNN_SIMD_CONST_F32(var, val) \
  static const xnn_simd_f32_t var =  \
      (xnn_simd_f32_t)((__f32x4){(val), (val), (val), (val)});

#define XNN_SIMD_CONST_F32_FROM_INT32(var, val) \
  static const xnn_simd_f32_t var =             \
      (xnn_simd_f32_t)((__u32x4){(val), (val), (val), (val)});


// Arithmetic operations.
static XNN_INLINE xnn_simd_f32_t xnn_zero_f32() {
  return wasm_f32x4_const_splat(0.0f);
}

static XNN_INLINE xnn_simd_f32_t xnn_add_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return wasm_f32x4_add(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_mul_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return wasm_f32x4_mul(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_sub_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return wasm_f32x4_sub(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_div_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return wasm_f32x4_div(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_abs_f32(xnn_simd_f32_t a) {
  return wasm_f32x4_abs(a);
}

static XNN_INLINE xnn_simd_f32_t xnn_neg_f32(xnn_simd_f32_t a) {
  return wasm_f32x4_neg(a);
}

static XNN_INLINE xnn_simd_f32_t xnn_round_f32(xnn_simd_f32_t a) {
  return wasm_f32x4_nearest(a);
}

// Logical operations.
static XNN_INLINE xnn_simd_f32_t xnn_and_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return wasm_v128_and(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_or_f32(xnn_simd_f32_t a,
                                            xnn_simd_f32_t b) {
  return wasm_v128_or(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_xor_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return wasm_v128_xor(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_sll_f32(xnn_simd_f32_t a, uint8_t bits) {
  return wasm_i32x4_shl(a, bits);
}

static XNN_INLINE xnn_simd_f32_t xnn_srl_f32(xnn_simd_f32_t a, uint8_t bits) {
  return wasm_u32x4_shr(a, bits);
}

static XNN_INLINE xnn_simd_f32_t xnn_sra_f32(xnn_simd_f32_t a, uint8_t bits) {
  return wasm_i32x4_shr(a, bits);
}

static XNN_INLINE xnn_simd_f32_t xnn_cmpeq_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b) {
  return wasm_f32x4_eq(a, b);
}

// Special functions.
#define XNN_SIMD_HAVE_RCP_F32 0
#define XNN_SIMD_HAVE_RSQRT_F32 0

// Load/store operations.
static XNN_INLINE xnn_simd_f32_t xnn_loadu_f32(const float* ptr) {
  return wasm_v128_load(ptr);
}

static XNN_INLINE xnn_simd_f32_t xnn_load_f32(const float* ptr) {
  return wasm_v128_load(ptr);
}

static XNN_INLINE void xnn_storeu_f32(float* ptr, xnn_simd_f32_t v) {
  wasm_v128_store(ptr, v);
}

static XNN_INLINE void xnn_store_f32(float* ptr, xnn_simd_f32_t v) {
  wasm_v128_store(ptr, v);
}

static XNN_INLINE xnn_simd_f32_t xnn_set1_f32(float v) {
  return wasm_f32x4_splat(v);
}

// Tail load/store operations.
static XNN_INLINE xnn_simd_f32_t
xnn_load_tail_f32(const float* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_f32);
  return wasm_v128_load(input);
}

static XNN_INLINE xnn_simd_f32_t xnn_load_tail_safe_f32(const float* input,
                                                        size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_f32);

  XNN_ALIGN(16) float padded[4];
  float* dst = padded;
  switch (num_elements) {
    case 3:
      *dst++ = *input++;
    case 2:
      *dst++ = *input++;
    default:
      *dst++ = *input++;
  }
  return wasm_v128_load(padded);
}

static XNN_INLINE void xnn_store_tail_f32(float* output, xnn_simd_f32_t v,
                                          size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_f32);

  if (num_elements & 2) {
    wasm_v128_store64_lane(output, v, 0);
    v = wasm_i64x2_shuffle(v, v, 1, 1);
    output += 2;
  }
  if (num_elements & 1) {
    wasm_v128_store32_lane(output, v, 0);
  }
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_F32_WASMSIMD_BASE_H_
