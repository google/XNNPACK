// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_S8_SCALAR_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_S8_SCALAR_H_

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"

// SIMD vector type for s8 using SCALAR.
typedef int8_t xnn_simd_s8_t;
#define xnn_simd_size_s8 1
#define xnn_simd_log2_size_s8 0
#define xnn_simd_bytes_s8 (xnn_simd_size_s8 * sizeof(int8_t))

#define XNN_SIMD_CONST_S8(var, val) static const xnn_simd_s8_t var = val;

// Arithmetic operations.
static XNN_INLINE xnn_simd_s8_t xnn_add_s8(xnn_simd_s8_t a, xnn_simd_s8_t b) {
  return a + b;
}

static XNN_INLINE xnn_simd_s8_t xnn_loadu_s8(const int8_t *ptr) { return *ptr; }

static XNN_INLINE xnn_simd_s8_t xnn_load_s8(const int8_t *ptr) { return *ptr; }

static XNN_INLINE void xnn_storeu_s8(int8_t *ptr, xnn_simd_s8_t v) { *ptr = v; }

static XNN_INLINE void xnn_store_s8(int8_t *ptr, xnn_simd_s8_t v) { *ptr = v; }

static XNN_INLINE xnn_simd_s8_t xnn_set1_s8(int8_t v) { return v; }

static XNN_INLINE xnn_simd_s8_t xnn_set1_or_load_s8(const int8_t *v) {
  return *v;
}

// Tail load/store operations.
static XNN_INLINE xnn_simd_s8_t xnn_load_tail_s8(const int8_t *input,
                                                 size_t num_elements) {
  return *input;
}

static XNN_INLINE void xnn_store_tail_s8(int8_t *output, xnn_simd_s8_t v,
                                         size_t num_elements) {
  *output = v;
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_S8_SCALAR_H_
