// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_U32_NEON_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_U32_NEON_H_

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>

#include <arm_neon.h>

#include "xnnpack/common.h"

// SIMD vector type for u32 using NEON.
typedef uint32x4_t xnn_simd_u32_t;
#define xnn_simd_size_u32 4
#define xnn_simd_log2_size_u32 2
#define xnn_simd_bytes_u32 (xnn_simd_size_u32 * sizeof(uint32_t))

#define XNN_SIMD_CONST_S32(var, val) const uint32x4_t var = vdupq_n_u32(val);

// Arithmetic operations.
static XNN_INLINE xnn_simd_u32_t xnn_mul_u32(xnn_simd_u32_t a,
                                             xnn_simd_u32_t b) {
  return vmulq_u32(a, b);
}

static XNN_INLINE xnn_simd_u32_t xnn_max_u32(xnn_simd_u32_t a,
                                             xnn_simd_u32_t b) {
  return vmaxq_u32(a, b);
}

static XNN_INLINE xnn_simd_u32_t xnn_min_u32(xnn_simd_u32_t a,
                                             xnn_simd_u32_t b) {
  return vminq_u32(a, b);
}

static XNN_INLINE xnn_simd_u32_t xnn_sub_u32(xnn_simd_u32_t a,
                                             xnn_simd_u32_t b) {
  return vsubq_u32(a, b);
}

static XNN_INLINE float32x4_t xnn_subw_f32_u32(xnn_simd_u32_t a,
                                               xnn_simd_u32_t b) {
#if XNN_ARCH_ARM64
  int64x2_t result_low = vreinterpretq_s64_u64(
      vsubw_u32(vmovl_u32(vget_low_u32(a)), vget_low_u32(b)));
  int64x2_t result_high = vreinterpretq_s64_u64(
      vsubw_u32(vmovl_u32(vget_high_u32(a)), vget_high_u32(b)));
    return vcombine_f32(
      vcvt_f32_f64(vcvtq_f64_s64(result_low)),
      vcvt_f32_f64(vcvtq_f64_s64(result_high)));
#else
  const uint32x4_t mask = vcgtq_u32(a, b);
  const float32x4_t variant1 = vcvtq_f32_u32(vsubq_u32(a, b));
  const float32x4_t variant2 = vcvtq_f32_u32(vsubq_u32(b, a));
  const float32x4_t sign = vbslq_f32(mask, vdupq_n_f32(1), vdupq_n_f32(-1));
  return vmulq_f32(vbslq_f32(mask, variant1, variant2), sign);
#endif
}

// Load/store operations.
static XNN_INLINE xnn_simd_u32_t xnn_loadu_u32(const uint32_t* ptr) {
  return vld1q_u32(ptr);
}

static XNN_INLINE xnn_simd_u32_t xnn_load_u32(const uint32_t* ptr) {
  return vld1q_u32(ptr);
}

static XNN_INLINE void xnn_storeu_u32(uint32_t* ptr, xnn_simd_u32_t v) {
  vst1q_u32(ptr, v);
}

static XNN_INLINE void xnn_store_u32(uint32_t* ptr, xnn_simd_u32_t v) {
  vst1q_u32(ptr, v);
}

static XNN_INLINE xnn_simd_u32_t xnn_set1_u32(uint32_t v) {
  return vld1q_dup_u32(&v);
}

static XNN_INLINE xnn_simd_u32_t xnn_set1_or_load_u32(const uint32_t* v) {
  return vld1q_dup_u32(v);
}

// Tail load/store operations.
static XNN_INLINE xnn_simd_u32_t
xnn_load_tail_u32(const uint32_t* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_u32);
  return vld1q_u32(input);
}

static XNN_INLINE void xnn_store_tail_u32(uint32_t* output, xnn_simd_u32_t v,
                                          size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_u32);

  uint32x2_t v_low = vget_low_u32(v);
  if (num_elements & 2) {
    vst1_u32(output, v_low);
    output += 2;
    v_low = vget_high_u32(v);
  }
  if (num_elements & 1) {
    vst1_lane_u32(output, v_low, 0);
  }
}

// Conversion operations.

static XNN_INLINE float32x4_t xnn_cvt_f32_u32(xnn_simd_u32_t a) {
  return vcvtq_f32_u32(a);
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_U32_NEON_H_
