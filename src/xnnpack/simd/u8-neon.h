// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_U8_NEON_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_U8_NEON_H_

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/unaligned.h"

// SIMD vector type for u8 using NEON.
typedef uint8x16_t xnn_simd_u8_t;
#define xnn_simd_size_u8 16
#define xnn_simd_log2_size_u8 0
#define xnn_simd_bytes_u8 (xnn_simd_size_u8 * sizeof(uint8_t))

#define XNN_SIMD_CONST_U8(var, val) const xnn_simd_u8_t var = vdupq_n_u8(val);

// Arithmetic operations.

static XNN_INLINE xnn_simd_u8_t xnn_add_u8(xnn_simd_u8_t a, xnn_simd_u8_t b) {
  return vaddq_u8(a, b);
}

static XNN_INLINE xnn_simd_u8_t xnn_max_u8(xnn_simd_u8_t a, xnn_simd_u8_t b) {
  return vmaxq_u8(a, b);
}

static XNN_INLINE uint8_t xnn_horizontal_max_u8(xnn_simd_u8_t a) {
  uint8x8_t max0 = vpmax_u8(vget_low_u8(a), vget_high_u8(a));
  max0 = vpmax_u8(max0, max0);
  max0 = vpmax_u8(max0, max0);
  max0 = vpmax_u8(max0, max0);

  return vget_lane_u8(max0, 0);
}

static XNN_INLINE uint8_t xnn_horizontal_min_u8(xnn_simd_u8_t a) {
  uint8x8_t min0 = vpmin_u8(vget_low_u8(a), vget_high_u8(a));
  min0 = vpmin_u8(min0, min0);
  min0 = vpmin_u8(min0, min0);
  min0 = vpmin_u8(min0, min0);

  return vget_lane_u8(min0, 0);
}

static XNN_INLINE xnn_simd_u8_t xnn_min_u8(xnn_simd_u8_t a, xnn_simd_u8_t b) {
  return vminq_u8(a, b);
}

static XNN_INLINE xnn_simd_u8_t xnn_xor_u8(xnn_simd_u8_t a, xnn_simd_u8_t b) {
  return veorq_u8(a, b);
}

// Load/store operations.

static XNN_INLINE xnn_simd_u8_t xnn_loadu_u8(const uint8_t* ptr) {
  return vld1q_u8(ptr);
}

static XNN_INLINE xnn_simd_u8_t xnn_load_u8(const uint8_t* ptr) {
  return vld1q_u8(ptr);
}

static XNN_INLINE void xnn_storeu_u8(uint8_t* ptr, xnn_simd_u8_t v) {
  vst1q_u8(ptr, v);
}

static XNN_INLINE void xnn_store_u8(uint8_t* ptr, xnn_simd_u8_t v) {
  vst1q_u8(ptr, v);
}

static XNN_INLINE xnn_simd_u8_t xnn_set1_u8(uint8_t v) { return vdupq_n_u8(v); }

// Tail load/store operations.

static XNN_INLINE xnn_simd_u8_t
xnn_load_tail_u8(const uint8_t* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_u8);
  return vld1q_u8(input);
}

static XNN_INLINE xnn_simd_u8_t xnn_load_tail_safe_u8(const uint8_t* input,
                                                      size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_u8);

  XNN_ALIGN(16) uint8_t padded[16];
  uint8_t* d = &padded[0];
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
  return vld1q_u8(&padded[0]);
}

static XNN_INLINE void xnn_store_tail_u8(uint8_t* output, xnn_simd_u8_t v,
                                         size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_u8);

  uint8x8_t v_lo = vget_low_u8(v);
  if (num_elements & (8 * sizeof(uint8_t))) {
    vst1_u8(output, v_lo);
    output += 8;
    v_lo = vget_high_u8(v);
  }
  if (num_elements & (4 * sizeof(uint8_t))) {
    unaligned_store_u32(output, vget_lane_u32(vreinterpret_u32_u8(v_lo), 0));
    output += 4;
    v_lo = vext_u8(v_lo, v_lo, 4);
  }
  if (num_elements & (2 * sizeof(uint8_t))) {
    unaligned_store_u16(output, vget_lane_u16(vreinterpret_u16_u8(v_lo), 0));
    output += 2;
    v_lo = vext_u8(v_lo, v_lo, 2);
  }
  if (num_elements & (1 * sizeof(uint8_t))) {
    vst1_lane_u8(output, v_lo, 0);
  }
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_U8_NEON_H_
