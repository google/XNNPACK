// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_S8_NEON_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_S8_NEON_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include "xnnpack/common.h"

// SIMD vector type for s8 using NEON.
typedef int8x16_t xnn_simd_s8_t;
#define xnn_simd_size_s8 16
#define xnn_simd_log2_size_s8 4
#define xnn_simd_bytes_s8 (xnn_simd_size_s8 * sizeof(int8_t))

#define XNN_SIMD_CONST_S8(var, val) const int8x16_t var = vdupq_n_s8(val);

// Arithmetic operations.

static XNN_INLINE xnn_simd_s8_t xnn_add_s8(xnn_simd_s8_t a,
                                             xnn_simd_s8_t b) {
  return vaddq_s8(a, b);
}

static XNN_INLINE xnn_simd_s8_t xnn_max_s8(xnn_simd_s8_t a,
                                             xnn_simd_s8_t b) {
  return vmaxq_s8(a, b);
}

static XNN_INLINE xnn_simd_s8_t xnn_min_s8(xnn_simd_s8_t a,
                                             xnn_simd_s8_t b) {
  return vminq_s8(a, b);
}

static XNN_INLINE int16x8_t xnn_low_cvt_s8_s16(xnn_simd_s8_t a) {
  return vmovl_s8(vget_low_s8(a));
}

static XNN_INLINE int16x8_t xnn_high_cvt_s8_s16(xnn_simd_s8_t a) {
  return vmovl_s8(vget_high_s8(a));
}

// Load/store operations.
static XNN_INLINE xnn_simd_s8_t xnn_loadu_s8(const int8_t* ptr) {
  return vld1q_s8(ptr);
}

static XNN_INLINE xnn_simd_s8_t xnn_load_s8(const int8_t* ptr) {
  return vld1q_s8(ptr);
}

static XNN_INLINE void xnn_storeu_s8(int8_t* ptr, xnn_simd_s8_t v) {
  vst1q_s8(ptr, v);
}

static XNN_INLINE void xnn_store_s8(int8_t* ptr, xnn_simd_s8_t v) {
  vst1q_s8(ptr, v);
}

static XNN_INLINE xnn_simd_s8_t xnn_set1_s8(int8_t v) {
  return vld1q_dup_s8(&v);
}

static XNN_INLINE xnn_simd_s8_t xnn_set1_or_load_s8(const int8_t* v) {
  return vld1q_dup_s8(v);
}

// Tail load/store operations.
static XNN_INLINE xnn_simd_s8_t
xnn_load_tail_s8(const int8_t* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s8);
  return vld1q_s8(input);
}

static XNN_INLINE void xnn_store_tail_s8(int8_t* output, xnn_simd_s8_t v,
                                          size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s8);

  int8x8_t v_low = vget_low_s8(v);
  if (num_elements & 8) {
    vst1_s8(output, v_low);
    output += 8;
    v_low = vget_high_s8(v);
  }
  if (num_elements & 4) {
    vst1_lane_s32((void*) output, vreinterpret_s32_s8(v_low), 0);
    output += 4;
    v_low = vext_s8(v_low, v_low, 4);
  }
  if (num_elements & 2) {
    vst1_lane_s16((void*) output, vreinterpret_s16_s8(v_low), 0);
    output += 2;
    v_low = vext_s8(v_low, v_low, 2);
  }
  if (num_elements & 1) {
    vst1_lane_s8(output, v_low, 0);
  }
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_S8_NEON_H_
