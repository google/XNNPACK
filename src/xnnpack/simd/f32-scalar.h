// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_F32_SCALAR_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_F32_SCALAR_H_

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"


// SIMD vector type for f32 using SCALAR.
typedef float xnn_simd_f32_t;
#define xnn_simd_size_f32 1
#define xnn_simd_log2_size_f32 0
#define xnn_simd_bytes_f32 (xnn_simd_size_f32 * sizeof(float))

#define XNN_SIMD_CONST_F32(var, val) \
  static const xnn_simd_f32_t var = val;

#define XNN_SIMD_CONST_F32_FROM_INT32(var, val)             \
  static const int32_t _##var##_int_value = val; \
  const xnn_simd_f32_t var = *(const float *)&_##var##_int_value;

// Include the header for generic functions _after_ declaring the arch-specific
// types and sizes.
#include "xnnpack/simd/f32-generic-functions.h"

// Arithmetic operations.
static XNN_INLINE xnn_simd_f32_t xnn_zero_f32() { return 0.0f; }

static XNN_INLINE xnn_simd_f32_t xnn_add_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return a + b;
}

static XNN_INLINE xnn_simd_f32_t xnn_mul_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return a * b;
}

static XNN_INLINE xnn_simd_f32_t xnn_fmadd_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b,
                                               xnn_simd_f32_t c) {
  return (a * b) + c;
}

static XNN_INLINE xnn_simd_f32_t xnn_fnmadd_f32(xnn_simd_f32_t a,
                                                xnn_simd_f32_t b,
                                                xnn_simd_f32_t c) {
  return c - (a * b);
}

static XNN_INLINE xnn_simd_f32_t xnn_fmsub_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b,
                                               xnn_simd_f32_t c) {
  return (a * b) - c;
}

static XNN_INLINE xnn_simd_f32_t xnn_sub_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return a - b;
}

static XNN_INLINE xnn_simd_f32_t xnn_div_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return a / b;
}

static XNN_INLINE xnn_simd_f32_t xnn_max_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return a > b ? a : b;
}

static XNN_INLINE xnn_simd_f32_t xnn_min_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return a < b ? a : b;
}

static XNN_INLINE xnn_simd_f32_t xnn_abs_f32(xnn_simd_f32_t a) { return fabsf(a); }

static XNN_INLINE xnn_simd_f32_t xnn_neg_f32(xnn_simd_f32_t a) { return -a; }

// Logical operations.
static XNN_INLINE xnn_simd_f32_t xnn_and_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  const uint32_t res = *(const uint32_t *)&a & *(const uint32_t *)&b;
  return *(const xnn_simd_f32_t *)&res;
}

static XNN_INLINE xnn_simd_f32_t xnn_or_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  const uint32_t res = *(const uint32_t *)&a | *(const uint32_t *)&b;
  return *(const xnn_simd_f32_t *)&res;
}

static XNN_INLINE xnn_simd_f32_t xnn_xor_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  const uint32_t res = *(const uint32_t *)&a ^ *(const uint32_t *)&b;
  return *(const xnn_simd_f32_t *)&res;
}

static XNN_INLINE xnn_simd_f32_t xnn_sll_f32(xnn_simd_f32_t a, uint8_t bits) {
  const uint32_t res = *(uint32_t *)&a << bits;
  return *(const xnn_simd_f32_t *)&res;
}

static XNN_INLINE xnn_simd_f32_t xnn_srl_f32(xnn_simd_f32_t a, uint8_t bits) {
  const uint32_t res = *(uint32_t *)&a >> bits;
  return *(const xnn_simd_f32_t *)&res;
}

static XNN_INLINE xnn_simd_f32_t xnn_sra_f32(xnn_simd_f32_t a, uint8_t bits) {
  const int32_t res = *(int32_t *)&a >> bits;
  return *(const xnn_simd_f32_t *)&res;
}

static XNN_INLINE xnn_simd_f32_t xnn_cmpeq_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b) {
  XNN_SIMD_CONST_F32_FROM_INT32(ones, 0xFFFFFFFF)
  return a == b ? ones : 0.0f;
}

// Special functions.
#define XNN_SIMD_HAVE_RCP_F32 0
#define XNN_SIMD_HAVE_RSQRT_F32 0

static XNN_INLINE xnn_simd_f32_t xnn_getexp_f32(xnn_simd_f32_t a) {
  return xnn_generic_getexp_f32(a);
}

// Load/store operations.
static XNN_INLINE xnn_simd_f32_t xnn_loadu_f32(const float *ptr) { return *ptr; }

static XNN_INLINE xnn_simd_f32_t xnn_load_f32(const float *ptr) { return *ptr; }

static XNN_INLINE void xnn_storeu_f32(float *ptr, xnn_simd_f32_t v) { *ptr = v; }

static XNN_INLINE void xnn_store_f32(float *ptr, xnn_simd_f32_t v) { *ptr = v; }

static XNN_INLINE xnn_simd_f32_t xnn_set1_f32(float v) { return v; }

static XNN_INLINE xnn_simd_f32_t xnn_set1_or_load_f32(const float *v) { return *v; }

// Tail load/store operations.
static XNN_INLINE xnn_simd_f32_t xnn_load_tail_f32(const float *input,
                                                   size_t num_elements) {
  return *input;
}

static XNN_INLINE void xnn_store_tail_f32(float *output, xnn_simd_f32_t v,
                                          size_t num_elements) {
  *output = v;
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_F32_SCALAR_H_
