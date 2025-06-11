// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_S16_NEON_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_S16_NEON_H_

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"

// SIMD vector type for s16 using NEON.
typedef int16x8_t xnn_simd_s16_t;
#define xnn_simd_size_s16 8
#define xnn_simd_log2_size_s16 3
#define xnn_simd_bytes_s16 (xnn_simd_size_s16 * sizeof(int16_t))

#define XNN_SIMD_CONST_S16(var, val) const int16x8_t var = vdupq_n_s16(val);

// Arithmetic operations.

static XNN_INLINE xnn_simd_s16_t xnn_min_s16(xnn_simd_s16_t a,
                                             xnn_simd_s16_t b) {
  return vminq_s16(a, b);
}

static XNN_INLINE xnn_simd_s16_t xnn_max_s16(xnn_simd_s16_t a,
                                             xnn_simd_s16_t b) {
  return vmaxq_s16(a, b);
}

static XNN_INLINE xnn_simd_s16_t xnn_signcomplement_s16(xnn_simd_s16_t x) {
  XNN_SIMD_CONST_S16(nonsign_mask, 0x7FFF);
  return veorq_s16(vandq_s16(x, nonsign_mask), vshrq_n_s16(x, 15));
}

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

// Tail load/store operations.
static XNN_INLINE xnn_simd_s16_t
xnn_load_tail_s16(const int16_t* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s16);
  return vld1q_s16(input);
}

static XNN_INLINE xnn_simd_s16_t xnn_load_tail_safe_s16(const int16_t* input,
                                                        size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s16);

  XNN_ALIGN(16) int16_t padded[8];
  int16_t* d = &padded[0];
  switch (num_elements) {
    case 7:
      *d++ = *input++;
    case 6:
      *d++ = *input++;
    case 5:
      *d++ = *input++;
    case 4:
      *d++ = *input++;
    case 3:
      *d++ = *input++;
    case 2:
      *d++ = *input++;
    case 1:
      *d++ = *input++;
  }
  return vld1q_s16(&padded[0]);
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
    vst1_lane_s32((int32_t*)output, vreinterpret_s32_s16(v_low), 0);
    output += 2;
    v_low = vext_s16(v_low, v_low, 2);
  }
  if (num_elements & 1) {
    vst1_lane_s16(output, v_low, 0);
  }
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_S16_NEON_H_
