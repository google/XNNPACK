// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_U8_SCALAR_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_U8_SCALAR_H_

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"

// SIMD vector type for u8 using SCALAR.
typedef uint8_t xnn_simd_u8_t;
#define xnn_simd_size_u8 1
#define xnn_simd_log2_size_u8 0
#define xnn_simd_bytes_u8 (xnn_simd_size_u8 * sizeof(uint8_t))

#define XNN_SIMD_CONST_u8(var, val) static const xnn_simd_u8_t var = val;

// Arithmetic operations.
static XNN_INLINE xnn_simd_u8_t xnn_add_u8(xnn_simd_u8_t a, xnn_simd_u8_t b) {
  return a + b;
}

static XNN_INLINE xnn_simd_u8_t xnn_min_u8(xnn_simd_u8_t a, xnn_simd_u8_t b) {
  return a < b ? a : b;
}

static XNN_INLINE xnn_simd_u8_t xnn_max_u8(xnn_simd_u8_t a, xnn_simd_u8_t b) {
  return a < b ? b : a;
}

static XNN_INLINE uint8_t xnn_horizontal_min_u8(xnn_simd_u8_t a) { return a; }

static XNN_INLINE uint8_t xnn_horizontal_max_u8(xnn_simd_u8_t a) { return a; }

static XNN_INLINE xnn_simd_u8_t xnn_xor_u8(xnn_simd_u8_t a, xnn_simd_u8_t b) {
  return a ^ b;
}

static XNN_INLINE xnn_simd_u8_t xnn_loadu_u8(const uint8_t *ptr) {
  return *ptr;
}

static XNN_INLINE xnn_simd_u8_t xnn_load_u8(const uint8_t *ptr) { return *ptr; }

static XNN_INLINE void xnn_storeu_u8(uint8_t *ptr, xnn_simd_u8_t v) {
  *ptr = v;
}

static XNN_INLINE void xnn_store_u8(uint8_t *ptr, xnn_simd_u8_t v) { *ptr = v; }

static XNN_INLINE xnn_simd_u8_t xnn_set1_u8(uint8_t v) { return v; }

// Tail load/store operations.
static XNN_INLINE xnn_simd_u8_t xnn_load_tail_u8(const uint8_t *input,
                                                 size_t num_elements) {
  return *input;
}

static XNN_INLINE xnn_simd_u8_t xnn_load_tail_safe_u8(const uint8_t *input,
                                                      size_t num_elements) {
  return *input;
}

static XNN_INLINE void xnn_store_tail_u8(uint8_t *output, xnn_simd_u8_t v,
                                         size_t num_elements) {
  *output = v;
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_U8_SCALAR_H_
