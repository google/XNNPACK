// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_S32_SCALAR_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_S32_SCALAR_H_

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"


// SIMD vector type for s32 using SCALAR.
typedef int32_t xnn_simd_s32_t;
#define xnn_simd_size_s32 1
#define xnn_simd_log2_size_s32 0
#define xnn_simd_bytes_s32 (xnn_simd_size_s32 * sizeof(int32_t))

#define XNN_SIMD_CONST_S32(var, val) \
  static const xnn_simd_s32_t var = val;

// Arithmetic operations.
static XNN_INLINE xnn_simd_s32_t xnn_mul_s32(xnn_simd_s32_t a,
                                             xnn_simd_s32_t b) {
  return ((int64_t) a * (int64_t) b) & (((int64_t)1 << 32) - 1);
}

static XNN_INLINE xnn_simd_s32_t xnn_max_s32(xnn_simd_s32_t a,
                                             xnn_simd_s32_t b) {
  return (a > b) ? a : b;
}

static XNN_INLINE xnn_simd_s32_t xnn_min_s32(xnn_simd_s32_t a,
                                             xnn_simd_s32_t b) {
  return (a < b) ? a : b;
}

static XNN_INLINE xnn_simd_s32_t xnn_loadu_s32(const int32_t *ptr) { return *ptr; }

static XNN_INLINE xnn_simd_s32_t xnn_load_s32(const int32_t *ptr) { return *ptr; }

static XNN_INLINE void xnn_storeu_s32(int32_t *ptr, xnn_simd_s32_t v) { *ptr = v; }

static XNN_INLINE void xnn_store_s32(int32_t *ptr, xnn_simd_s32_t v) { *ptr = v; }

static XNN_INLINE xnn_simd_s32_t xnn_set1_s32(int32_t v) { return v; }

static XNN_INLINE xnn_simd_s32_t xnn_set1_or_load_s32(const int32_t *v) { return *v; }

// Tail load/store operations.
static XNN_INLINE xnn_simd_s32_t xnn_load_tail_s32(const int32_t *input,
                                                   size_t num_elements) {
  return *input;
}

static XNN_INLINE void xnn_store_tail_s32(int32_t *output, xnn_simd_s32_t v,
                                          size_t num_elements) {
  *output = v;
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_F32_SCALAR_H_
