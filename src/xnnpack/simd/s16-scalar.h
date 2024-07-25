// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_S16_SCALAR_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_S16_SCALAR_H_

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"


// SIMD vector type for s16 using SCALAR.
typedef int16_t xnn_simd_s16_t;
#define xnn_simd_size_s16 1
#define xnn_simd_log2_size_s16 0
#define xnn_simd_bytes_s16 (xnn_simd_size_s16 * sizeof(int16_t))

#define XNN_SIMD_CONST_S16(var, val) \
  static const xnn_simd_s16_t var = val;

// Arithmetic operations.

static XNN_INLINE xnn_simd_s16_t xnn_loadu_s16(const int16_t *ptr) { return *ptr; }

static XNN_INLINE xnn_simd_s16_t xnn_load_s16(const int16_t *ptr) { return *ptr; }

static XNN_INLINE void xnn_storeu_s16(int16_t *ptr, xnn_simd_s16_t v) { *ptr = v; }

static XNN_INLINE void xnn_store_s16(int16_t *ptr, xnn_simd_s16_t v) { *ptr = v; }

static XNN_INLINE xnn_simd_s16_t xnn_set1_s16(int16_t v) { return v; }

static XNN_INLINE xnn_simd_s16_t xnn_set1_or_load_s16(const int16_t *v) { return *v; }

// Tail load/store operations.
static XNN_INLINE xnn_simd_s16_t xnn_load_tail_s16(const int16_t *input,
                                                   size_t num_elements) {
  return *input;
}

static XNN_INLINE void xnn_store_tail_s16(int16_t *output, xnn_simd_s16_t v,
                                          size_t num_elements) {
  *output = v;
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_S16_SCALAR_H_
