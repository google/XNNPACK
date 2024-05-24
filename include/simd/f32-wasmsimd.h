// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef THIRD_PARTY_XNNPACK_INCLUDE_SIMD_F32_WASMSIMD_H_
#define THIRD_PARTY_XNNPACK_INCLUDE_SIMD_F32_WASMSIMD_H_

#include <assert.h>
#include <stddef.h>
#include <wasm_simd128.h>
#include <xnnpack/common.h>

// SIMD vector type for f32 using WASMSIMD.
typedef v128_t xnn_simd_f32_t;
#define xnn_simd_size_f32 4
static const size_t xnn_simd_log2_size_f32 = 2;
static const size_t xnn_simd_bytes_f32 = xnn_simd_size_f32 * sizeof(float);

#define xnn_simd_static_init_f32(var, val) \
  static const xnn_simd_f32_t var = wasm_f32x4_make((val), (val), (val), (val));

#define xnn_simd_static_init_i32(var, val) \
  static const v128_t var = wasm_u32x4_make((val), (val), (val), (val));

// Arithmetic operations.
XNN_INLINE xnn_simd_f32_t xnn_zero_f32() {
  return wasm_f32x4_const_splat(0.0f);
}

XNN_INLINE xnn_simd_f32_t xnn_add_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return wasm_f32x4_add(a, b);
}

XNN_INLINE xnn_simd_f32_t xnn_mul_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return wasm_f32x4_mul(a, b);
}

XNN_INLINE xnn_simd_f32_t xnn_fmadd_f32(xnn_simd_f32_t a, xnn_simd_f32_t b,
                                        xnn_simd_f32_t c) {
  return wasm_f32x4_add(wasm_f32x4_mul(a, b), c);
}

XNN_INLINE xnn_simd_f32_t xnn_fnmadd_f32(xnn_simd_f32_t a, xnn_simd_f32_t b,
                                         xnn_simd_f32_t c) {
  return wasm_f32x4_sub(c, wasm_f32x4_mul(a, b));
}

XNN_INLINE xnn_simd_f32_t xnn_sub_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return wasm_f32x4_sub(a, b);
}

XNN_INLINE xnn_simd_f32_t xnn_div_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return wasm_f32x4_div(a, b);
}

XNN_INLINE xnn_simd_f32_t xnn_max_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return wasm_f32x4_max(a, b);
}

XNN_INLINE xnn_simd_f32_t xnn_min_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return wasm_f32x4_min(a, b);
}

XNN_INLINE xnn_simd_f32_t xnn_abs_f32(xnn_simd_f32_t a) {
  return wasm_f32x4_abs(a);
}

XNN_INLINE xnn_simd_f32_t xnn_neg_f32(xnn_simd_f32_t a) {
  return wasm_f32x4_neg(a);
}

// Logical operations.
XNN_INLINE xnn_simd_f32_t xnn_and_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return wasm_v128_and(a, b);
}

XNN_INLINE xnn_simd_f32_t xnn_or_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return wasm_v128_or(a, b);
}

XNN_INLINE xnn_simd_f32_t xnn_xor_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return wasm_v128_xor(a, b);
}

// Special functions.
#define XNN_SIMD_HAVE_RCP_F32 0
#define XNN_SIMD_HAVE_RSQRT_F32 0

// Load/store operations.
XNN_INLINE xnn_simd_f32_t xnn_loadu_f32(const float* ptr) {
  return wasm_v128_load(ptr);
}

XNN_INLINE xnn_simd_f32_t xnn_load_f32(const float* ptr) {
  return wasm_v128_load(ptr);
}

XNN_INLINE void xnn_storeu_f32(float* ptr, xnn_simd_f32_t v) {
  wasm_v128_store(ptr, v);
}

XNN_INLINE void xnn_store_f32(float* ptr, xnn_simd_f32_t v) {
  wasm_v128_store(ptr, v);
}

XNN_INLINE xnn_simd_f32_t xnn_set1_f32(float v) { return wasm_f32x4_splat(v); }

XNN_INLINE xnn_simd_f32_t xnn_set1_or_load_f32(const float* v) {
  return wasm_f32x4_splat(*v);
}

// Tail load/store operations.
XNN_INLINE xnn_simd_f32_t xnn_load_tail_f32(const float* input,
                                            size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_f32);
  return wasm_v128_load(input);
}

XNN_INLINE void xnn_store_tail_f32(float* output, xnn_simd_f32_t v,
                                   size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_f32);

  if (num_elements & 2) {
    wasm_v128_store64_lane(output, v, 0);
    v = wasm_v64x2_shuffle(v, v, 1, 1);
    output += 2;
  }
  if (num_elements & 1) {
    wasm_v128_store32_lane(output, v, 0);
  }
}

#endif  // THIRD_PARTY_XNNPACK_INCLUDE_SIMD_F32_WASMSIMD_H_
