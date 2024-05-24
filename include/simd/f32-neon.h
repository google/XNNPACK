// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef THIRD_PARTY_XNNPACK_INCLUDE_SIMD_F32_NEON_H_
#define THIRD_PARTY_XNNPACK_INCLUDE_SIMD_F32_NEON_H_

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <xnnpack/common.h>

// SIMD vector type for f32 using NEON.
typedef float32x4_t xnn_simd_f32_t;
#define xnn_simd_size_f32 4
static const size_t xnn_simd_log2_size_f32 = 2;
static const size_t xnn_simd_bytes_f32 = xnn_simd_size_f32 * sizeof(float);

#define xnn_simd_static_init_f32(var, val) \
  static const xnn_simd_f32_t var = {(val), (val), (val), (val)};

#define xnn_simd_static_init_i32(var, val) \
  static const uint32x4_t var = {(val), (val), (val), (val)};

// Arithmetic operations.
XNN_INLINE xnn_simd_f32_t xnn_zero_f32() { return vdupq_n_f32(0.f); }

XNN_INLINE xnn_simd_f32_t xnn_add_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return vaddq_f32(a, b);
}

XNN_INLINE xnn_simd_f32_t xnn_mul_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return vmulq_f32(a, b);
}

XNN_INLINE xnn_simd_f32_t xnn_sub_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return vsubq_f32(a, b);
}

XNN_INLINE xnn_simd_f32_t xnn_fmadd_f32(xnn_simd_f32_t a, xnn_simd_f32_t b,
                                        xnn_simd_f32_t c) {
  return vmlaq_f32(c, a, b);
}

XNN_INLINE xnn_simd_f32_t xnn_fnmadd_f32(xnn_simd_f32_t a, xnn_simd_f32_t b,
                                         xnn_simd_f32_t c) {
  return vmlsq_f32(c, a, b);
}

XNN_INLINE xnn_simd_f32_t xnn_div_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
#if XNN_ARCH_ARM64
  return vdivq_f32(a, b);
#else
  float32x4_t rb = vrecpeq_f32(b);
  rb = vmulq_f32(rb, vrecpsq_f32(rb, b));
  rb = vmulq_f32(rb, vrecpsq_f32(rb, b));
  return vmulq_f32(a, rb);
#endif  // XNN_ARCH_ARM64
}

XNN_INLINE xnn_simd_f32_t xnn_max_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return vmaxq_f32(a, b);
}

XNN_INLINE xnn_simd_f32_t xnn_min_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return vminq_f32(a, b);
}

XNN_INLINE xnn_simd_f32_t xnn_abs_f32(xnn_simd_f32_t a) { return vabsq_f32(a); }

XNN_INLINE xnn_simd_f32_t xnn_neg_f32(xnn_simd_f32_t a) { return vnegq_f32(a); }

// Logical operations.
XNN_INLINE xnn_simd_f32_t xnn_and_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return vreinterpretq_f32_s32(
      vandq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b)));
}

XNN_INLINE xnn_simd_f32_t xnn_or_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return vreinterpretq_f32_s32(
      vorrq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b)));
}

XNN_INLINE xnn_simd_f32_t xnn_xor_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return vreinterpretq_f32_s32(
      veorq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b)));
}

// Special functions.
#define XNN_SIMD_HAVE_RCP_F32 1
#define XNN_SIMD_NUM_RCP_ITER_F32 2
XNN_INLINE xnn_simd_f32_t xnn_rcp_f32(xnn_simd_f32_t a) {
  return vrecpeq_f32(a);
}

#define XNN_SIMD_HAVE_RSQRT_F32 1
#define XNN_SIMD_NUM_RSQRT_ITER_F32 2
XNN_INLINE xnn_simd_f32_t xnn_rsqrt_f32(xnn_simd_f32_t a) {
  return vrsqrteq_f32(a);
}

// Load/store operations.
XNN_INLINE xnn_simd_f32_t xnn_loadu_f32(const float* ptr) {
  return vld1q_f32(ptr);
}

XNN_INLINE xnn_simd_f32_t xnn_load_f32(const float* ptr) {
  return vld1q_f32(ptr);
}

XNN_INLINE void xnn_storeu_f32(float* ptr, xnn_simd_f32_t v) {
  vst1q_f32(ptr, v);
}

XNN_INLINE void xnn_store_f32(float* ptr, xnn_simd_f32_t v) {
  vst1q_f32(ptr, v);
}

XNN_INLINE xnn_simd_f32_t xnn_set1_f32(float v) { return vld1q_dup_f32(&v); }

XNN_INLINE xnn_simd_f32_t xnn_set1_or_load_f32(const float* v) {
  return vld1q_dup_f32(v);
}

// Tail load/store operations.
XNN_INLINE xnn_simd_f32_t xnn_load_tail_f32(const float* input,
                                            size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_f32);
  return vld1q_f32(input);
}

XNN_INLINE void xnn_store_tail_f32(float* output, xnn_simd_f32_t v,
                                   size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_f32);

  float32x2_t v_low = vget_low_f32(v);
  if (num_elements & 2) {
    vst1_f32(output, v_low);
    output += 2;
    v_low = vget_high_f32(v);
  }
  if (num_elements & 1) {
    vst1_lane_f32(output, v_low, 0);
  }
}

#endif  // THIRD_PARTY_XNNPACK_INCLUDE_SIMD_F32_NEON_H_
