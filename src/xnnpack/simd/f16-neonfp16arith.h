// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef THIRD_PARTY_XNNPACK_SRC_XNNPACK_SIMD_F16_NEONFP16ARITH_H_
#define THIRD_PARTY_XNNPACK_SRC_XNNPACK_SIMD_F16_NEONFP16ARITH_H_

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"

// SIMD vector type for f16 using NEON.
typedef float16x8_t xnn_simd_f16_t;
#define xnn_simd_size_f16 8
#define xnn_simd_log2_size_f16 3
#define xnn_simd_bytes_f16 (xnn_simd_size_f16 * sizeof(xnn_float16))

#define XNN_SIMD_CONST_F16(var, val) const float16x8_t var = vdupq_n_f16(val);

#define XNN_SIMD_CONST_F16_FROM_INT16(var, val) \
  const float16x8_t var = vreinterpretq_f16_u16(vdupq_n_u16(val))

#if XNN_HAVE_FLOAT16
#define XNN_SIMD_CONST_F16_FROM_FLOAT(var, val) \
  const float16x8_t var = vdupq_n_f16(xnn_float16_from_float(val))
#else
#define XNN_SIMD_CONST_F16_FROM_FLOAT(var, val) \
  XNN_SIMD_CONST_F16_FROM_INT16(                \
      var, xnn_float16_to_bits(xnn_float16_from_float(val)))
#endif  // XNN_HAVE_FLOAT16

// Whether or not this architecture has native fused multiply-add support.
#if __ARM_FEATURE_FMA
#define XNN_SIMD_HAS_NATIVE_FMA 1
#else
#define XNN_SIMD_HAS_NATIVE_FMA 0
#endif  // __ARM_FEATURE_FMA

// The following wrapper is defined as a macro since `bits` needs to be a
// compile-time constant.
#define XNN_SIMD_SHIFTS_ARE_MACROS 1
#define xnn_sll_f16(a, bits) \
  vreinterpretq_f16_u16(vshlq_n_u16(vreinterpretq_u16_f16(a), bits))

// The following wrapper is defined as a macro since `bits` needs to be a
// compile-time constant.
#define xnn_srl_f16(a, bits) \
  vreinterpretq_f16_u16(vshrq_n_u16(vreinterpretq_u16_f16(a), bits))

// Arithmetic operations.
static XNN_INLINE xnn_simd_f16_t xnn_zero_f16() {
  return vreinterpretq_f16_u16(vdupq_n_u16(0));
}

static XNN_INLINE xnn_simd_f16_t xnn_add_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return vaddq_f16(a, b);
}

static XNN_INLINE xnn_simd_f16_t xnn_mul_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return vmulq_f16(a, b);
}

static XNN_INLINE xnn_simd_f16_t xnn_sub_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return vsubq_f16(a, b);
}

static XNN_INLINE xnn_simd_f16_t xnn_fmadd_f16(xnn_simd_f16_t a,
                                               xnn_simd_f16_t b,
                                               xnn_simd_f16_t c) {
#if __ARM_FEATURE_FMA
  return vfmaq_f16(c, a, b);
#else
  return vaddq_f16(vmulq_f16(a, b), c);
#endif  // __ARM_FEATURE_FMA
}

static XNN_INLINE xnn_simd_f16_t xnn_fnmadd_f16(xnn_simd_f16_t a,
                                                xnn_simd_f16_t b,
                                                xnn_simd_f16_t c) {
#if __ARM_FEATURE_FMA
  return vfmsq_f16(c, a, b);
#else
  return vsubq_f16(c, vmulq_f16(a, b));
#endif  // __ARM_FEATURE_FMA
}

static XNN_INLINE xnn_simd_f16_t xnn_fmsub_f16(xnn_simd_f16_t a,
                                               xnn_simd_f16_t b,
                                               xnn_simd_f16_t c) {
#if __ARM_FEATURE_FMA
  return vfmaq_f16(vnegq_f16(c), a, b);
#else
  return vsubq_f16(vmulq_f16(a, b), c);
#endif  // __ARM_FEATURE_FMA
}

static XNN_INLINE xnn_simd_f16_t xnn_div_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
#if XNN_ARCH_ARM64
  return vdivq_f16(a, b);
#else
  float16x8_t rb = vrecpeq_f16(b);
  rb = vmulq_f16(rb, vrecpsq_f16(rb, b));
  rb = vmulq_f16(rb, vrecpsq_f16(rb, b));
  return vmulq_f16(vmulq_f16(a, rb), vrecpsq_f16(rb, b));
#endif  // XNN_ARCH_ARM64
}

static XNN_INLINE xnn_simd_f16_t xnn_max_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return vmaxq_f16(a, b);
}

static XNN_INLINE xnn_simd_f16_t xnn_min_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return vminq_f16(a, b);
}

static XNN_INLINE xnn_simd_f16_t xnn_abs_f16(xnn_simd_f16_t a) {
  return vabsq_f16(a);
}

static XNN_INLINE xnn_simd_f16_t xnn_neg_f16(xnn_simd_f16_t a) {
  return vnegq_f16(a);
}

static XNN_INLINE xnn_simd_f16_t xnn_round_f16(xnn_simd_f16_t a) {
  return vrndnq_f16(a);
}

// Logical operations.
static XNN_INLINE xnn_simd_f16_t xnn_and_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return vreinterpretq_f16_s16(
      vandq_s16(vreinterpretq_s16_f16(a), vreinterpretq_s16_f16(b)));
}

static XNN_INLINE xnn_simd_f16_t xnn_or_f16(xnn_simd_f16_t a,
                                            xnn_simd_f16_t b) {
  return vreinterpretq_f16_s16(
      vorrq_s16(vreinterpretq_s16_f16(a), vreinterpretq_s16_f16(b)));
}

static XNN_INLINE xnn_simd_f16_t xnn_xor_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return vreinterpretq_f16_s16(
      veorq_s16(vreinterpretq_s16_f16(a), vreinterpretq_s16_f16(b)));
}

static XNN_INLINE xnn_simd_f16_t xnn_cmpeq_f16(xnn_simd_f16_t a,
                                               xnn_simd_f16_t b) {
  return vreinterpretq_f16_u16(vceqq_f16(a, b));
}

// Special functions.
#define XNN_SIMD_HAVE_RCP_F16 1
#define XNN_SIMD_NUM_RCP_ITER_F16 2
static XNN_INLINE xnn_simd_f16_t xnn_rcp_f16(xnn_simd_f16_t a) {
  return vrecpeq_f16(a);
}

#define XNN_SIMD_HAVE_RSQRT_F16 1
#define XNN_SIMD_NUM_RSQRT_ITER_F16 2
static XNN_INLINE xnn_simd_f16_t xnn_rsqrt_f16(xnn_simd_f16_t a) {
  return vrsqrteq_f16(a);
}

// Load/store operations.
//
// Note that since MSVC doesn't support the `vld1q_f16` and `vst1q_f16`
// intrinsics, we convert to `u16` for load/store.
static XNN_INLINE xnn_simd_f16_t xnn_loadu_f16(const xnn_float16 *ptr) {
  return vreinterpretq_f16_u16(vld1q_u16((const uint16_t *)ptr));
}

static XNN_INLINE xnn_simd_f16_t xnn_load_f16(const xnn_float16 *ptr) {
  return vreinterpretq_f16_u16(vld1q_u16((const uint16_t *)ptr));
}

static XNN_INLINE void xnn_storeu_f16(xnn_float16 *ptr, xnn_simd_f16_t v) {
  vst1q_u16((uint16_t *)ptr, vreinterpretq_u16_f16(v));
}

static XNN_INLINE void xnn_store_f16(xnn_float16 *ptr, xnn_simd_f16_t v) {
  vst1q_u16((uint16_t *)ptr, vreinterpretq_u16_f16(v));
}

static XNN_INLINE xnn_simd_f16_t xnn_set1_f16(xnn_float16 v) {
  return vreinterpretq_f16_u16(vld1q_dup_u16((const uint16_t *)&v));
}

// Tail load/store operations.
static XNN_INLINE xnn_simd_f16_t
xnn_load_tail_f16(const xnn_float16 *input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_f16);
  return vreinterpretq_f16_u16(vld1q_u16((const uint16_t *)input));
}

static xnn_simd_f16_t xnn_load_tail_safe_f16(const xnn_float16 *input,
                                             size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_f16);

  XNN_ALIGN(16) xnn_float16 padded[8];
  xnn_float16 *dst = padded;
  switch (num_elements) {
    case 7:
      *dst++ = *input++;
    case 6:
      *dst++ = *input++;
    case 5:
      *dst++ = *input++;
    case 4:
      *dst++ = *input++;
    case 3:
      *dst++ = *input++;
    case 2:
      *dst++ = *input++;
    default:
      *dst++ = *input++;
  }
  return vreinterpretq_f16_u16(vld1q_u16((const uint16_t *)&padded[0]));
}

static XNN_INLINE void xnn_store_tail_f16(xnn_float16 *output, xnn_simd_f16_t v,
                                          size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_f16);

  float16x4_t v_low = vget_low_f16(v);
  if (num_elements & 4) {
    vst1_u16((uint16_t *)output, vreinterpret_u16_f16(v_low));
    output += 4;
    v_low = vget_high_f16(v);
  }
  if (num_elements & 2) {
    vst1_lane_s32((int32_t *)output, vreinterpret_s32_f16(v_low), 0);
    output += 2;
    v_low = vext_f16(v_low, v_low, 2);
  }
  if (num_elements & 1) {
    vst1_lane_u16((uint16_t *)output, vreinterpret_u16_f16(v_low), 0);
  }
}

#endif  // THIRD_PARTY_XNNPACK_SRC_XNNPACK_SIMD_F16_NEONFP16ARITH_H_
