// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_S16_NEON_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_S16_NEON_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include "xnnpack/common.h"

// SIMD vector type for s16 using NEON.
typedef int16x8_t xnn_simd_s16_t;
#define xnn_simd_size_s16 8
#define xnn_simd_log2_size_s16 3
#define xnn_simd_bytes_s16 (xnn_simd_size_s16 * sizeof(int16_t))

#define XNN_SIMD_CONST_S16(var, val) const int16x8_t var = vdupq_n_s16(val);

// Arithmetic operations.

// Load/store operations.
static XNN_INLINE xnn_simd_s16_t xnn_loadu_s16(const int16_t* ptr) {
  return vld1q_s16(ptr);
}

static XNN_INLINE xnn_simd_s16_t xnn_load_s16(const int16_t* ptr) {
  return vld1q_s16(ptr);
}

static XNN_INLINE void xnn_storeu_s16(int16_t* ptr, xnn_simd_s16_t v) {
  vst1q_s16(ptr, v);
}

static XNN_INLINE void xnn_store_s16(int16_t* ptr, xnn_simd_s16_t v) {
  vst1q_s16(ptr, v);
}

static XNN_INLINE xnn_simd_s16_t xnn_set1_s16(int16_t v) {
  return vld1q_dup_s16(&v);
}

static XNN_INLINE xnn_simd_s16_t xnn_set1_or_load_s16(const int16_t* v) {
  return vld1q_dup_s16(v);
}

// Tail load/store operations.
static XNN_INLINE xnn_simd_s16_t
xnn_load_tail_s16(const int16_t* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s16);
  return vld1q_s16(input);
}

static XNN_INLINE void xnn_store_tail_s16(int16_t* output, xnn_simd_s16_t v,
                                          size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s16);

  int16x4_t v_low = vget_low_s16(v);
  if (num_elements & 4) {
    vst1_s16(output, v_low);
    output += 4;
    v_low = vget_high_s16(v);
  }
  if (num_elements & 2) {
    vst1_lane_s32((void*) output, vreinterpret_s32_s16(v_low), 0);
    output += 2;
    v_low = vext_s16(v_low, v_low, 2);
  }
  if (num_elements & 1) {
    vst1_lane_s16(output, v_low, 0);
  }
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_S16_NEON_H_
