// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_F32_NEON_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_F32_NEON_H_

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/common.h"

// SIMD vector type for f32 using NEON.
typedef float32x4_t xnn_simd_f32_t;
#define xnn_simd_size_f32 4
#define xnn_simd_log2_size_f32 2
#define xnn_simd_bytes_f32 (xnn_simd_size_f32 * sizeof(float))

#define XNN_SIMD_CONST_F32(var, val) const float32x4_t var = vdupq_n_f32(val);

#define XNN_SIMD_CONST_F32_FROM_INT32(var, val) \
  const float32x4_t var = vreinterpretq_f32_u32(vdupq_n_u32(val));

// Whether or not this architecture has native fused multiply-add support.
#if __ARM_FEATURE_FMA
#define XNN_SIMD_HAS_NATIVE_FMA 1
#else
#define XNN_SIMD_HAS_NATIVE_FMA 0
#endif  // __ARM_FEATURE_FMA

// The following wrapper is defined as a macro since `bits` needs to be a
// compile-time constant.
#define XNN_SIMD_SHIFTS_ARE_MACROS 1
#define xnn_sll_f32(a, bits) \
  vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_f32(a), bits))

// The following wrapper is defined as a macro since `bits` needs to be a
// compile-time constant.
#define xnn_srl_f32(a, bits) \
  vreinterpretq_f32_u32(vshrq_n_u32(vreinterpretq_u32_f32(a), bits))

// The following wrapper is defined as a macro since `bits` needs to be a
// compile-time constant.
#define xnn_sra_f32(a, bits) \
  vreinterpretq_f32_s32(vshrq_n_s32(vreinterpretq_s32_f32(a), bits))

// Arithmetic operations.
static XNN_INLINE xnn_simd_f32_t xnn_zero_f32() { return vdupq_n_f32(0.f); }

static XNN_INLINE xnn_simd_f32_t xnn_add_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return vaddq_f32(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_mul_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return vmulq_f32(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_sub_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return vsubq_f32(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_fmadd_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b,
                                               xnn_simd_f32_t c) {
#if __ARM_FEATURE_FMA
  return vfmaq_f32(c, a, b);
#else
  return vaddq_f32(vmulq_f32(a, b), c);
#endif  // __ARM_FEATURE_FMA
}

static XNN_INLINE xnn_simd_f32_t xnn_fnmadd_f32(xnn_simd_f32_t a,
                                                xnn_simd_f32_t b,
                                                xnn_simd_f32_t c) {
#if __ARM_FEATURE_FMA
  return vfmsq_f32(c, a, b);
#else
  return vsubq_f32(c, vmulq_f32(a, b));
#endif  // __ARM_FEATURE_FMA
}

static XNN_INLINE xnn_simd_f32_t xnn_fmsub_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b,
                                               xnn_simd_f32_t c) {
#if __ARM_FEATURE_FMA
  return vfmaq_f32(vnegq_f32(c), a, b);
#else
  return vsubq_f32(vmulq_f32(a, b), c);
#endif  // __ARM_FEATURE_FMA
}

static XNN_INLINE xnn_simd_f32_t xnn_div_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
#if XNN_ARCH_ARM64
  return vdivq_f32(a, b);
#else
  float32x4_t rb = vrecpeq_f32(b);
  rb = vmulq_f32(rb, vrecpsq_f32(rb, b));
  rb = vmulq_f32(rb, vrecpsq_f32(rb, b));
  return vmulq_f32(vmulq_f32(a, rb), vrecpsq_f32(rb, b));
#endif  // XNN_ARCH_ARM64
}

static XNN_INLINE xnn_simd_f32_t xnn_max_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return vmaxq_f32(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_min_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return vminq_f32(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_abs_f32(xnn_simd_f32_t a) {
  return vabsq_f32(a);
}

static XNN_INLINE xnn_simd_f32_t xnn_neg_f32(xnn_simd_f32_t a) {
  return vnegq_f32(a);
}

// Logical operations.
static XNN_INLINE xnn_simd_f32_t xnn_and_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return vreinterpretq_f32_s32(
      vandq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b)));
}

static XNN_INLINE xnn_simd_f32_t xnn_or_f32(xnn_simd_f32_t a,
                                            xnn_simd_f32_t b) {
  return vreinterpretq_f32_s32(
      vorrq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b)));
}

static XNN_INLINE xnn_simd_f32_t xnn_xor_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return vreinterpretq_f32_s32(
      veorq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b)));
}

static XNN_INLINE xnn_simd_f32_t xnn_not_f32(xnn_simd_f32_t a) {
  return vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(a)));
}

static XNN_INLINE xnn_simd_f32_t xnn_cmpeq_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b) {
  return vreinterpretq_f32_u32(vceqq_f32(a, b));
}

static XNN_INLINE xnn_simd_f32_t xnn_round_f32(xnn_simd_f32_t a) {
#if defined(__ARM_ARCH) && __ARM_ARCH < 8
  // Any input larger than 2^23 is already an integer value since its fractional
  // bits will no longer fit in the mantissa. We create a filter for these that
  // also catches all non-finite values in `a` (compares with NaN are always
  // `false`).
  XNN_SIMD_CONST_F32(vmax_non_int_val, 8388608.0f);  // 2^23.
  const xnn_simd_f32_t vfilter =
      vreinterpretq_f32_u32(vcaltq_f32(a, vmax_non_int_val));

  // Create a vector of `0.5f` with the same sign as the entries of `a`.
  XNN_SIMD_CONST_F32(vhalf, 0.5f);
  XNN_SIMD_CONST_F32(vsign_mask, -0.0);
  const xnn_simd_f32_t vsigned_half =
      xnn_or_f32(xnn_and_f32(a, vsign_mask), vhalf);
  const xnn_simd_f32_t vresult =
      vcvtq_f32_s32(vcvtq_s32_f32(xnn_add_f32(a, vsigned_half)));

  // Apply the non-finite value filter to replace any non-finite input with `a`.
  return xnn_or_f32(xnn_and_f32(vfilter, vresult),
                    xnn_and_f32(xnn_not_f32(vfilter), a));
#else
  return vrndnq_f32(a);
#endif  // defined(__ARM_ARCH) && __ARM_ARCH == 7
}

// Special functions.
#define XNN_SIMD_HAVE_RCP_F32 1
#define XNN_SIMD_NUM_RCP_ITER_F32 2
static XNN_INLINE xnn_simd_f32_t xnn_rcp_f32(xnn_simd_f32_t a) {
  return vrecpeq_f32(a);
}

#define XNN_SIMD_HAVE_RSQRT_F32 1
#define XNN_SIMD_NUM_RSQRT_ITER_F32 2
static XNN_INLINE xnn_simd_f32_t xnn_rsqrt_f32(xnn_simd_f32_t a) {
  return vrsqrteq_f32(a);
}

// Load/store operations.
static XNN_INLINE xnn_simd_f32_t xnn_loadu_f32(const float* ptr) {
  return vld1q_f32(ptr);
}

static XNN_INLINE xnn_simd_f32_t xnn_load_f32(const float* ptr) {
  return vld1q_f32(ptr);
}

static XNN_INLINE void xnn_storeu_f32(float* ptr, xnn_simd_f32_t v) {
  vst1q_f32(ptr, v);
}

static XNN_INLINE void xnn_store_f32(float* ptr, xnn_simd_f32_t v) {
  vst1q_f32(ptr, v);
}

static XNN_INLINE xnn_simd_f32_t xnn_set1_f32(float v) {
  return vld1q_dup_f32(&v);
}

// Tail load/store operations.
static XNN_INLINE xnn_simd_f32_t
xnn_load_tail_f32(const float* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_f32);
  return vld1q_f32(input);
}

// TODO: Use direct load of 1,2 or 3 floats
// Consider clearing pad values to 0
static XNN_INLINE xnn_simd_f32_t xnn_load_tail_safe_f32(const float* input,
                                                        size_t num_elements) {
  assert(num_elements <= xnn_simd_size_f32);

  XNN_ALIGN(16) float padded[4];
  float* dst = padded;
  switch (num_elements) {
    case 4:
      *dst++ = *input++;
    case 3:
      *dst++ = *input++;
    case 2:
      *dst++ = *input++;
    case 1:
      *dst++ = *input++;
    default: {
    }
  }
  return vld1q_f32(padded);
}

static XNN_INLINE void xnn_store_tail_f32(float* output, xnn_simd_f32_t v,
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

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_F32_NEON_H_
