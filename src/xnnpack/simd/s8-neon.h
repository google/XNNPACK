// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_S8_NEON_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_S8_NEON_H_

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/unaligned.h"

// SIMD vector type for s8 using NEON.
typedef int8x16_t xnn_simd_s8_t;
#define xnn_simd_size_s8 16
#define xnn_simd_log2_size_s8 0
#define xnn_simd_bytes_s8 (xnn_simd_size_s8 * sizeof(int8_t))

#define XNN_SIMD_CONST_U8(var, val) const xnn_simd_s8_t var = vdupq_n_s8(val);

// Arithmetic operations.

static XNN_INLINE xnn_simd_s8_t xnn_add_s8(xnn_simd_s8_t a, xnn_simd_s8_t b) {
  return vaddq_s8(a, b);
}

static XNN_INLINE xnn_simd_s8_t xnn_max_s8(xnn_simd_s8_t a, xnn_simd_s8_t b) {
  return vmaxq_s8(a, b);
}

static XNN_INLINE xnn_simd_s8_t xnn_min_s8(xnn_simd_s8_t a, xnn_simd_s8_t b) {
  return vminq_s8(a, b);
}

static XNN_INLINE int8_t xnn_horizontal_max_s8(xnn_simd_s8_t a) {
  int8x8_t max0 = vpmax_s8(vget_low_s8(a), vget_high_s8(a));
  max0 = vpmax_s8(max0, max0);
  max0 = vpmax_s8(max0, max0);
  max0 = vpmax_s8(max0, max0);

  return vget_lane_s8(max0, 0);
}

static XNN_INLINE int8_t xnn_horizontal_min_s8(xnn_simd_s8_t a) {
  int8x8_t min0 = vpmin_s8(vget_low_s8(a), vget_high_s8(a));
  min0 = vpmin_s8(min0, min0);
  min0 = vpmin_s8(min0, min0);
  min0 = vpmin_s8(min0, min0);

  return vget_lane_s8(min0, 0);
}

static XNN_INLINE xnn_simd_s8_t xnn_xor_s8(xnn_simd_s8_t a, xnn_simd_s8_t b) {
  return veorq_s8(a, b);
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

static XNN_INLINE xnn_simd_s8_t xnn_set1_s8(int8_t v) { return vdupq_n_s8(v); }

// Tail load/store operations.

static XNN_INLINE xnn_simd_s8_t
xnn_load_tail_s8(const int8_t* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s8);
  return vld1q_s8(input);
}

static XNN_INLINE xnn_simd_s8_t xnn_load_tail_safe_s8(const int8_t* input,
                                                      size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s8);

  XNN_ALIGN(16) int8_t padded[16];
  int8_t* d = &padded[0];
  switch (num_elements) {
    case 15:
      *d++ = *input++;
    case 14:
      *d++ = *input++;
    case 13:
      *d++ = *input++;
    case 12:
      *d++ = *input++;
    case 11:
      *d++ = *input++;
    case 10:
      *d++ = *input++;
    case 9:
      *d++ = *input++;
    case 8:
      *d++ = *input++;
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
  return vld1q_s8(&padded[0]);
}

static XNN_INLINE void xnn_store_tail_s8(int8_t* output, xnn_simd_s8_t v,
                                         size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s8);

  int8x8_t v_lo = vget_low_s8(v);
  if (num_elements & (8 * sizeof(int8_t))) {
    vst1_s8(output, v_lo);
    output += 8;
    v_lo = vget_high_s8(v);
  }
  if (num_elements & (4 * sizeof(int8_t))) {
    unaligned_store_u32(output, vget_lane_s32(vreinterpret_s32_s8(v_lo), 0));
    output += 4;
    v_lo = vext_s8(v_lo, v_lo, 4);
  }
  if (num_elements & (2 * sizeof(int8_t))) {
    unaligned_store_u16(output, vget_lane_s16(vreinterpret_s16_s8(v_lo), 0));
    output += 2;
    v_lo = vext_s8(v_lo, v_lo, 2);
  }
  if (num_elements & (1 * sizeof(int8_t))) {
    vst1_lane_s8(output, v_lo, 0);
  }
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_S8_NEON_H_
