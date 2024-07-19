// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_S32_NEON_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_S32_NEON_H_

#include <assert.h>
#include <stddef.h>

#include <arm_neon.h>

#include "xnnpack/common.h"

// SIMD vector type for s32 using NEON.
typedef int32x4_t xnn_simd_s32_t;
#define xnn_simd_size_s32 4
#define xnn_simd_log2_size_s32 2
#define xnn_simd_bytes_s32 (xnn_simd_size_s32 * sizeof(int32_t))

#define XNN_SIMD_CONST_S32(var, val) const int32x4_t var = vdupq_n_s32(val);

// Arithmetic operations.
static XNN_INLINE xnn_simd_s32_t xnn_mul_s32(xnn_simd_s32_t a,
                                             xnn_simd_s32_t b) {
  return vmulq_s32(a, b);
}

static XNN_INLINE xnn_simd_s32_t xnn_max_s32(xnn_simd_s32_t a,
                                             xnn_simd_s32_t b) {
  return vmaxq_s32(a, b);
}

static XNN_INLINE xnn_simd_s32_t xnn_min_s32(xnn_simd_s32_t a,
                                             xnn_simd_s32_t b) {
  return vminq_s32(a, b);
}

// Load/store operations.
static XNN_INLINE xnn_simd_s32_t xnn_loadu_s32(const int32_t* ptr) {
  return vld1q_s32(ptr);
}

static XNN_INLINE xnn_simd_s32_t xnn_load_s32(const int32_t* ptr) {
  return vld1q_s32(ptr);
}

static XNN_INLINE void xnn_storeu_s32(int32_t* ptr, xnn_simd_s32_t v) {
  vst1q_s32(ptr, v);
}

static XNN_INLINE void xnn_store_s32(int32_t* ptr, xnn_simd_s32_t v) {
  vst1q_s32(ptr, v);
}

static XNN_INLINE xnn_simd_s32_t xnn_set1_s32(int32_t v) {
  return vld1q_dup_s32(&v);
}

static XNN_INLINE xnn_simd_s32_t xnn_set1_or_load_s32(const int32_t* v) {
  return vld1q_dup_s32(v);
}

// Tail load/store operations.
static XNN_INLINE xnn_simd_s32_t
xnn_load_tail_s32(const int32_t* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s32);
  return vld1q_s32(input);
}

static XNN_INLINE void xnn_store_tail_s32(int32_t* output, xnn_simd_s32_t v,
                                          size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s32);

  int32x2_t v_low = vget_low_s32(v);
  if (num_elements & 2) {
    vst1_s32(output, v_low);
    output += 2;
    v_low = vget_high_s32(v);
  }
  if (num_elements & 1) {
    vst1_lane_s32(output, v_low, 0);
  }
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_S32_NEON_H_
