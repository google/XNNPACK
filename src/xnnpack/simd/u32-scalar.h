// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_U32_SCALAR_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_U32_SCALAR_H_

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"


// SIMD vector type for u32 using SCALAR.
typedef uint32_t xnn_simd_u32_t;
#define xnn_simd_size_u32 1
#define xnn_simd_log2_size_u32 0
#define xnn_simd_bytes_u32 (xnn_simd_size_u32 * sizeof(uint32_t))

#define XNN_SIMD_CONST_U32(var, val) \
  static const xnn_simd_u32_t var = val;

// Arithmetic operations.
static XNN_INLINE xnn_simd_u32_t xnn_mul_u32(xnn_simd_u32_t a,
                                             xnn_simd_u32_t b) {
  return ((uint64_t) a * (uint64_t) b) & (((uint64_t)1 << 32) - 1);
}

static XNN_INLINE xnn_simd_u32_t xnn_max_u32(xnn_simd_u32_t a,
                                             xnn_simd_u32_t b) {
  return (a > b) ? a : b;
}

static XNN_INLINE xnn_simd_u32_t xnn_min_u32(xnn_simd_u32_t a,
                                             xnn_simd_u32_t b) {
  return (a < b) ? a : b;
}

static XNN_INLINE xnn_simd_u32_t xnn_sub_u32(xnn_simd_u32_t a,
                                             xnn_simd_u32_t b) {
  return a - b;
}

static XNN_INLINE float xnn_subw_f32_u32(xnn_simd_u32_t a,
                                         xnn_simd_u32_t b) {
  return (float) ((int64_t) a - (int64_t) b);
}

static XNN_INLINE xnn_simd_u32_t xnn_loadu_u32(const uint32_t *ptr) { return *ptr; }

static XNN_INLINE xnn_simd_u32_t xnn_load_u32(const uint32_t *ptr) { return *ptr; }

static XNN_INLINE void xnn_storeu_u32(uint32_t *ptr, xnn_simd_u32_t v) { *ptr = v; }

static XNN_INLINE void xnn_store_u32(uint32_t *ptr, xnn_simd_u32_t v) { *ptr = v; }

static XNN_INLINE xnn_simd_u32_t xnn_set1_u32(uint32_t v) { return v; }

static XNN_INLINE xnn_simd_u32_t xnn_set1_or_load_u32(const uint32_t *v) { return *v; }

// Tail load/store operations.
static XNN_INLINE xnn_simd_u32_t xnn_load_tail_u32(const uint32_t *input,
                                                   size_t num_elements) {
  return *input;
}

static XNN_INLINE void xnn_store_tail_u32(uint32_t *output, xnn_simd_u32_t v,
                                          size_t num_elements) {
  *output = v;
}

// Conversion operations.
static XNN_INLINE float xnn_cvt_f32_u32(xnn_simd_u32_t a) {
  return (float) a;
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_U32_SCALAR_H_
