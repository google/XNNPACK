// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef THIRD_PARTY_XNNPACK_INCLUDE_SIMD_F32_SCALAR_H_
#define THIRD_PARTY_XNNPACK_INCLUDE_SIMD_F32_SCALAR_H_

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <xnnpack/common.h>

// SIMD vector type for f32 using SCALAR.
typedef float xnn_simd_f32_t;
#define xnn_simd_size_f32 1
static const size_t xnn_simd_log2_size_f32 = 0;
static const size_t xnn_simd_bytes_f32 = xnn_simd_size_f32 * sizeof(float);

#define xnn_simd_static_init_f32(var, val) \
  static const xnn_simd_f32_t var = val;

#define xnn_simd_static_init_i32(var, val) static const __m128i var = val;

// Arithmetic operations.
XNN_INLINE xnn_simd_f32_t xnn_zero_f32() { return 0.0f; }

XNN_INLINE xnn_simd_f32_t xnn_add_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return a + b;
}

XNN_INLINE xnn_simd_f32_t xnn_mul_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return a * b;
}

XNN_INLINE xnn_simd_f32_t xnn_fmadd_f32(xnn_simd_f32_t a, xnn_simd_f32_t b,
                                        xnn_simd_f32_t c) {
  return (a * b) + c;
}

XNN_INLINE xnn_simd_f32_t xnn_fnmadd_f32(xnn_simd_f32_t a, xnn_simd_f32_t b,
                                         xnn_simd_f32_t c) {
  return c - (a * b);
}

XNN_INLINE xnn_simd_f32_t xnn_sub_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return a - b;
}

XNN_INLINE xnn_simd_f32_t xnn_div_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return a / b;
}

XNN_INLINE xnn_simd_f32_t xnn_max_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return a > b ? a : b;
}

XNN_INLINE xnn_simd_f32_t xnn_min_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return a < b ? a : b;
}

XNN_INLINE xnn_simd_f32_t xnn_abs_f32(xnn_simd_f32_t a) { return fabsf(a); }

XNN_INLINE xnn_simd_f32_t xnn_neg_f32(xnn_simd_f32_t a) { return -a; }

// Logical operations.
XNN_INLINE xnn_simd_f32_t xnn_and_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  const uint32_t res = *(const uint32_t *)&a & *(const uint32_t *)&a;
  return *(const xnn_simd_f32_t *)&res;
}

XNN_INLINE xnn_simd_f32_t xnn_or_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  const uint32_t res = *(const uint32_t *)&a | *(const uint32_t *)&a;
  return *(const xnn_simd_f32_t *)&res;
}

XNN_INLINE xnn_simd_f32_t xnn_xor_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  const uint32_t res = *(const uint32_t *)&a ^ *(const uint32_t *)&a;
  return *(const xnn_simd_f32_t *)&res;
}

// Special functions.
#define XNN_SIMD_HAVE_RCP_F32 0
#define XNN_SIMD_HAVE_RSQRT_F32 0

// Load/store operations.
XNN_INLINE xnn_simd_f32_t xnn_loadu_f32(const float *ptr) { return *ptr; }

XNN_INLINE xnn_simd_f32_t xnn_load_f32(const float *ptr) { return *ptr; }

XNN_INLINE void xnn_storeu_f32(float *ptr, xnn_simd_f32_t v) { *ptr = v; }

XNN_INLINE void xnn_store_f32(float *ptr, xnn_simd_f32_t v) { *ptr = v; }

XNN_INLINE xnn_simd_f32_t xnn_set1_f32(float v) { return v; }

XNN_INLINE xnn_simd_f32_t xnn_set1_or_load_f32(const float *v) { return *v; }

// Tail load/store operations.
XNN_INLINE xnn_simd_f32_t xnn_load_tail_f32(const float *input,
                                            size_t num_elements) XNN_OOB_READS {
  return *input;
}

XNN_INLINE void xnn_store_tail_f32(float *output, xnn_simd_f32_t v,
                                   size_t num_elements) {
  *output = v;
}

#endif  // THIRD_PARTY_XNNPACK_INCLUDE_SIMD_F32_SCALAR_H_
